import json

import numpy as np
import torch
from cv2.gapi import kernel
from pywt.data import camera
import cv2

from lerf.data.utils.feature_dataloader import FeatureDataloader
from lerf.encoders.image_encoder import BaseImageEncoder
from tqdm import tqdm
import plotly.graph_objects as go

from lerf.data.utils.embeddings_directory import Scene_Graph_Nerf_Module
from lerf.encoders.openclip_encoder import (OpenCLIPNetwork,
                                        OpenCLIPNetworkConfig)
from nerfstudio.cameras.cameras import Cameras
from jaxtyping import Float
from torch import Tensor



class PatchEmbeddingDataloader(FeatureDataloader):
    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        model: BaseImageEncoder,
        image_list: torch.Tensor = None,
        depths: torch.Tensor = None,
        cameras: Cameras = None,
        cache_path: str = None,
        dataparser_scale: float = None,
        applied_transform: Float[Tensor, "3 4"] = None,
        scene_graph: Scene_Graph_Nerf_Module = None
    ):
        assert "tile_ratio" in cfg
        assert "stride_ratio" in cfg
        assert "image_shape" in cfg
        assert "model_name" in cfg

        self.kernel_size = int(cfg["image_shape"][0] * cfg["tile_ratio"])
        self.stride = int(self.kernel_size * cfg["stride_ratio"])
        self.padding = self.kernel_size // 2
        self.center_x = (
            (self.kernel_size - 1) / 2
            - self.padding
            + self.stride
            * np.arange(
                np.floor((cfg["image_shape"][0] + 2 * self.padding - (self.kernel_size - 1) - 1) / self.stride + 1)
            )
        )
        self.center_y = (
            (self.kernel_size - 1) / 2
            - self.padding
            + self.stride
            * np.arange(
                np.floor((cfg["image_shape"][1] + 2 * self.padding - (self.kernel_size - 1) - 1) / self.stride + 1)
            )
        )
        self.center_x = torch.from_numpy(self.center_x).half()
        self.center_y = torch.from_numpy(self.center_y).half()
        self.start_x = self.center_x[0].float()
        self.start_y = self.center_y[0].float()

        self.model = model
        self.embed_size = self.model.embedding_dim

        self.SceneGraph = None

        self.cameras = cameras

        self.dataparser_scale = dataparser_scale
        self.applied_transform = applied_transform

        self.SceneGraph = scene_graph
        self.depths=depths

        super().__init__(cfg, device, image_list, cache_path)

    def load(self):
        cache_info_path = self.cache_path.with_suffix(".info")
        if not cache_info_path.exists():
            raise FileNotFoundError
        with open(cache_info_path, "r") as f:
            cfg = json.loads(f.read())
        if cfg != self.cfg:
            raise ValueError("Config mismatch")
        self.data = torch.from_numpy(np.load(self.cache_path)).half()

    def create(self, image_list):
        assert self.model is not None, "model must be provided to generate features"
        assert image_list is not None, "image_list must be provided to generate features"

        unfold_func = torch.nn.Unfold(
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        ).to(self.device)

        img_embeds = []
        index = 0
        print(f"images length: {image_list.shape[0]}")
        print(self.depths.shape)
        for img in tqdm(image_list, desc="Embedding images", leave=False):
            depth = self.depths[index].permute(2,1,0).squeeze()
            img_embeds.append(self._embed_clip_tiles(img.unsqueeze(0),depth, unfold_func, index))
            index = index + 1
            
        self.data = torch.from_numpy(np.stack(img_embeds)).half()

    def __call__(self, img_points):
        # img_points: (B, 3) # (img_ind, x, y) (img_ind, row, col)
        # return: (B, 512)
        img_points = img_points.cpu()
        img_ind, img_points_x, img_points_y = img_points[:, 0], img_points[:, 1], img_points[:, 2]

        x_ind = torch.floor((img_points_x - (self.start_x)) / self.stride).long()
        y_ind = torch.floor((img_points_y - (self.start_y)) / self.stride).long()
        return self._interp_inds(img_ind, x_ind, y_ind, img_points_x, img_points_y)

    def _interp_inds(self, img_ind, x_ind, y_ind, img_points_x, img_points_y):
        img_ind = img_ind.to(self.data.device)  # self.data is on cpu to save gpu memory, hence this line
        topleft = self.data[img_ind, x_ind, y_ind].to(self.device)
        topright = self.data[img_ind, x_ind + 1, y_ind].to(self.device)
        botleft = self.data[img_ind, x_ind, y_ind + 1].to(self.device)
        botright = self.data[img_ind, x_ind + 1, y_ind + 1].to(self.device)

        x_stride = self.stride
        y_stride = self.stride
        right_w = ((img_points_x - (self.center_x[x_ind])) / x_stride).to(self.device)  # .half()
        top = torch.lerp(topleft, topright, right_w[:, None])
        bot = torch.lerp(botleft, botright, right_w[:, None])

        bot_w = ((img_points_y - (self.center_y[y_ind])) / y_stride).to(self.device)  # .half()
        return torch.lerp(top, bot, bot_w[:, None])

    def _embed_clip_tiles(self, image, depth, unfold_func, index):
        # image augmentation: slow-ish (0.02s for 600x800 image per augmentation)
        aug_imgs = torch.cat([image])

        tiles = unfold_func(aug_imgs).permute(2, 0, 1).reshape(-1, 3, self.kernel_size, self.kernel_size).to("cuda")
        print(f"tiles shape: {tiles.shape}")

        # Assume aug_imgs is on the same device as the returned patch centers.
        _, _, H, W = aug_imgs.shape

        print(f"image shape: {aug_imgs.shape}")

        kernel_size = self.kernel_size
        stride = self.stride
        padding = self.padding

        # Calculate padded image dimensions
        padded_H = H + 2 * padding
        padded_W = W + 2 * padding

        # Compute number of patches along height and width (as used in unfolding)
        num_patches_y = (padded_H - kernel_size) // stride + 1
        num_patches_x = (padded_W - kernel_size) // stride + 1

        # Create 1D arrays for the top-left coordinates (in pixels) of each patch.
        # Multiply by stride because patches are sampled every `stride` pixels.
        x_indices = torch.arange(num_patches_x, device=aug_imgs.device, dtype=torch.float32)
        y_indices = torch.arange(num_patches_y, device=aug_imgs.device, dtype=torch.float32)

        # The center of a patch is its top-left coordinate plus half the kernel size.
        x_centers = -padding + x_indices * stride + kernel_size // 2
        y_centers = -padding + y_indices * stride + kernel_size // 2

        # Create a 2D grid (meshgrid) of these centers.
        # Using indexing='xy' ensures the first coordinate corresponds to x (columns)
        grid_x, grid_y = torch.meshgrid(x_centers, y_centers, indexing='xy')
        
        # Flatten the grids and stack them as (x, y) pairs.
        patch_centers = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

        print(f"patch centers shape: {patch_centers.shape}")

        assert(patch_centers.shape[0] == tiles.shape[0])

        #rays = self.cameras.generate_rays(camera_indices=index, coords=patch_centres)
        c2w = self.cameras[index].camera_to_worlds
        camera_origin = c2w[:, 3]

        embeddings = []

        origins = []
        directions = []

        ray_size = patch_centers.shape[0]
        special_mask = torch.zeros(ray_size, dtype=torch.bool)

        counter=0

        for i in range(ray_size):
            #transform ray origins and directions to original space
            inv_transform = torch.linalg.inv(
                torch.cat(
                    (
                        self.applied_transform,
                        torch.tensor([[0, 0, 0, 1]], dtype=self.applied_transform.dtype, device=self.applied_transform.device),
                    ),
                    0,
                )
            )

            origin = camera_origin / self.dataparser_scale
            #direction_ray = rays.directions[i] 

            #calculate ray direction
            fx = self.cameras[0].fx.item()
            fy = self.cameras[0].fy.item()
            cx = self.cameras[0].cx.item()
            cy = self.cameras[0].cy.item()

            #generate intrinsic matrix
            K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]])
            pixel_homg = torch.cat([patch_centers[i], torch.tensor([1.0])], dim=0)
            direction = np.matmul(np.linalg.inv(K), pixel_homg)
            #direction_norm = direction / np.linalg.norm(direction)
            #convert to tensor
            direction_tensor = torch.tensor(direction, dtype=torch.float32, device=self.applied_transform.device)
            #normalise the direction tensor
            direction_tensor = direction_tensor / torch.norm(direction_tensor)
            #normalise direction using z value
            #direction_tensor = direction_tensor / direction_tensor[2]

            #direction_tensor_homg = torch.cat([direction_tensor, torch.tensor([0.0])], dim=0)

            x_ray = patch_centers[i][0].item()
            y_ray = patch_centers[i][1].item()

            x_ray = min(x_ray, W - 1)
            y_ray = min(y_ray, H - 1)
            ray_radius = 5

            # # Define the patch size (3x3) and compute the offset
            # patch_size = 5
            # offset = patch_size // 2  # for 3x3, offset is 1

            # # Pad the image with a border of width 'offset'
            # # mode='edge' replicates the border pixels
            # depth_padded = torch.nn.functional.pad(depth.unsqueeze(0), (offset, offset, offset, offset), mode='replicate')

            # # Adjust the center pixel coordinates for the padded image
            # i_p, j_p = int(x_ray) + offset, int(y_ray) + offset

            # # Extract the patch from the padded image
            # patch = depth_padded[:, i_p - offset : i_p + offset + 1, j_p - offset : j_p + offset + 1]


            # found=None
            # nonzero_pixel=None
            # pos=None

            # tolerance = 0.15

            # depth_val = None
            # for col in range(patch_size):
            #     for row in range(patch_size):
            #         pixel = patch[:, col, row].item()
            #         if (pixel != 0):
            #             nonzero_pixel = pixel
            #             pos = (row, col)
            #             depth_val = pixel + tolerance
            #             found = True
            #         break
            #     if found:
            #         break
            # if (found is None):
            #     depth_val = 0

            #get depth
            #depth_val = depth[int(x_ray), int(y_ray)]

            ray_orig_homg = torch.cat([origin, torch.tensor([1.0])], dim=0)
            #ray_direction_homg = torch.cat([direction_ray, torch.tensor([0.0])], dim=0)

            transformed_origin = np.matmul(inv_transform, ray_orig_homg)

            #convert from homogenous to non-homogenous
            transformed_origin = transformed_origin[:3] / transformed_origin[3]
            #transformed_direction = transformed_direction[:3] / transformed_direction[3]

            #convert from opengl to opencv convention
            transformed_origin = torch.Tensor([transformed_origin[1], transformed_origin[0], -transformed_origin[2]])
            #transformed_direction = torch.Tensor([transformed_direction[1], transformed_direction[0], -transformed_direction[2]])

            #calculate the 3d point
            #point = transformed_origin + (depth_val * direction_tensor)


            #apply the extrinsic matrix
            c2w = self.cameras[index].camera_to_worlds
            extrinsic = torch.cat(
                    (
                        c2w,
                        torch.tensor([[0, 0, 0, 1]], dtype=self.applied_transform.dtype, device=self.applied_transform.device),
                    ),
                    0,
            )
            #scale the extrinsic matrix by the dataparser scale
            extrinsic[:3, 3] = extrinsic[:3, 3] / self.dataparser_scale
            extrinsic = torch.matmul(inv_transform, extrinsic)
            extrinsic=extrinsic.detach().cpu().numpy()
            extrinsic = np.linalg.inv(extrinsic)
            #convert the extrinsic matrix back to opencv convention
            # Reverse operation 3: multiply row 2 by -1
            extrinsic[2, :] *= -1
            # Reverse operation 2: swap rows 0 and 1
            extrinsic = extrinsic[np.array([1, 0, 2, 3]), :]
            # Reverse operation 1: multiply elements in rows 0-2 and columns 1-2 by -1
            extrinsic[0:3, 1:3] *= -1


            R = extrinsic[:3, :3]
            t = extrinsic[:3, 3]

            transformed_direction = np.matmul(R, direction_tensor)

            origins.append(transformed_origin)
            directions.append(transformed_direction)

            hit_info = self.SceneGraph.ray_intersection_normal(transformed_origin, transformed_direction)

            if (hit_info['hit']):
                special_mask[i] = True

                rvec, _ = cv2.Rodrigues(R)
                tvec = t.reshape(3, 1)# ensure t is a column vector

                box_index = hit_info["index"]
                embedding = self.SceneGraph.graph_embeddings[box_index]
                descriptor = self.SceneGraph.descriptors[box_index]
                embeddings.append(embedding)

                # extract bounding box information
                bounds = hit_info["bounding_box"]
                box_min = bounds[0]
                box_max = bounds[1]
                # Example 3D bounding box corners (8 points)
                corners = np.array([
                    [box_min[0], box_min[1], box_min[2]],
                    [box_min[0], box_min[1], box_max[2]],
                    [box_min[0], box_max[1], box_min[2]],
                    [box_min[0], box_max[1], box_max[2]],
                    [box_max[0], box_min[1], box_min[2]],
                    [box_max[0], box_min[1], box_max[2]],
                    [box_max[0], box_max[1], box_min[2]],
                    [box_max[0], box_max[1], box_max[2]]
                ], dtype=np.float32)



                # project the corners to the image plane
               
                corners, _ = cv2.projectPoints(corners, rvec, tvec, K, distCoeffs=None)
                img_points = np.int32(corners).reshape(-1, 2)

                # Draw the 3D bounding box by connecting the projected points
                # Define the connections (edges) of the bounding box:
                edges = [
                            (0, 2), (2, 3), (3, 1), (1, 0),  # Back face (Corrected)
                            (4, 6), (6, 7), (7, 5), (5, 4),  # Front face (Corrected)
                            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges (Correct)
                        ]


                
                img_np = aug_imgs.squeeze().cpu().numpy().transpose(1, 2, 0)  # Convert tensor to numpy array
                # Ensure the numpy array is contiguous and of type uint8
                rgb = np.ascontiguousarray((img_np * 255).astype(np.uint8))

                # Draw edges on the image
                for start, end in edges:
                    pt1 = tuple(img_points[start])
                    pt2 = tuple(img_points[end])
                    cv2.line(rgb, pt1, pt2, (0, 255, 0), 2)
                
                cv2.circle(rgb, (int(x_ray), int(y_ray)), ray_radius, (255, 0, 0), 2)
                rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
                # font
                font = cv2.FONT_HERSHEY_SIMPLEX

                # org
                org = (50, 50)

                # fontScale
                fontScale = 1
 
                # Blue color in BGR
                color = (255, 0, 0)

                # Line thickness of 2 px
                thickness = 2
 
                # Using cv2.putText() method
                rgb = cv2.putText(rgb, descriptor, org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)

                #only write tenth intersection to avoid a large amount of images
                if counter%10 == 0:
                    cv2.imwrite(f"/home/paperspace/code/lerf-scene-graph/outputs/outputs/bounding_boxes/image_{index}_ray_{i}.png", rgb)
                #cv2.imwrite(f"/home/paperspace/code/lerf-scene-graph/outputs/outputs/bounding_boxes/image_{index}_ray_{i}.png", rgb)
                counter=counter+1
            else:
                embeddings.append([])

        origins = torch.cat(origins).view(-1, 3)
        directions = torch.cat(directions).view(-1, 3)

        #mask rays that intersect
        # Separate rays into special and normal
        intersect_origins = origins[special_mask].reshape(-1, 3)
        normal_origins = origins[~special_mask].reshape(-1, 3)

        intersect_directions = directions[special_mask].reshape(-1, 3)
        normal_directions = directions[~special_mask].reshape(-1, 3)

        # Create the lines tensor (each ray represented by 2 consecutive points)
        lines = torch.empty((normal_origins.shape[0] * 2, 3))
        lines[0::2] = normal_origins
        lines[1::2] = normal_origins + normal_directions
       
        line_intersect = torch.empty((intersect_origins.shape[0] * 2, 3))
        line_intersect[0::2] = intersect_origins
        line_intersect[1::2] = intersect_origins + intersect_directions

        # Create separate traces with different colors
        # trace_normal = go.Scatter3d(
        #     x=lines[:, 0].numpy(),
        #     y=lines[:, 1].numpy(),
        #     z=lines[:, 2].numpy(),
        #     mode="lines",
        #     line=dict(color="lightblue", width=1),
        #     name="Normal Rays"
        # )

        # trace_special = go.Scatter3d(
        #     x=line_intersect[:, 0].numpy(),
        #     y=line_intersect[:, 1].numpy(),
        #     z=line_intersect[:, 2].numpy(),
        #     mode="lines",
        #     line=dict(color="red", width=2),
        #     name="Special Rays"
        # )

        # fig = go.Figure(data=[trace_normal, trace_special])

        # lines = torch.empty((origins.shape[0] * 2, 3))
        # lines[0::2] = origins
        # lines[1::2] = origins + directions
    
        # fig.update_layout(
        #     scene=dict(
        #         xaxis=dict(title="x", showspikes=False),
        #         yaxis=dict(title="y", showspikes=False),
        #         zaxis=dict(title="z", showspikes=False),
        #     ),
        #     margin=dict(r=0, b=10, l=0, t=10),
        #     hovermode=False,
        # )

        # traces = self.SceneGraph.draw_bboxes()
        # for trace in traces:
        #     fig.add_trace(trace)
        # fig.show()

        with torch.no_grad():
            clip_embeds = self.model.encode_image(tiles)
            #clip_embeds /= clip_embeds.norm(dim=-1, keepdim=True)

            for i, embed in enumerate(clip_embeds):
                embeddings[i].append(embed)


            averages = [torch.stack(sublist, dim=0).mean(dim=0) for sublist in embeddings]

            #text_embedding = torch.cat(embeddings, dim=1)
            #text_embedding = torch.cat(text_embedding, dim=0)
            #clip_embeds = torch.mean(text_embedding, dim=1)

            clip_embeds = torch.stack(averages, dim=0)

       

        clip_embeds = clip_embeds.reshape((self.center_x.shape[0], self.center_y.shape[0], -1))
        clip_embeds = torch.concat((clip_embeds, clip_embeds[:, [-1], :]), dim=1)
        clip_embeds = torch.concat((clip_embeds, clip_embeds[[-1], :, :]), dim=0)
        return clip_embeds.detach().cpu().numpy()
