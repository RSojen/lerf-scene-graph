import json

import numpy as np
import torch
from cv2.gapi import kernel
from pywt.data import camera

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
        for img in tqdm(image_list, desc="Embedding images", leave=False):
            img_embeds.append(self._embed_clip_tiles(img.unsqueeze(0), unfold_func, index))
            torch.cuda.empty_cache()
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

    def _embed_clip_tiles(self, image, unfold_func, index):
        # image augmentation: slow-ish (0.02s for 600x800 image per augmentation)
        aug_imgs = torch.cat([image])

        tiles = unfold_func(aug_imgs).permute(2, 0, 1).reshape(-1, 3, self.kernel_size, self.kernel_size).to("cuda")
        print(tiles.shape)

        # Extract input dimensions
        N, C, H, W = aug_imgs.shape

        padding = self.padding
        stride = self.stride
        kernel_size = self.kernel_size

        # Compute output grid dimensions for the unfolding operation
        H_out = (H + 2 * padding - kernel_size) // stride + 1
        W_out = (W + 2 * padding - kernel_size) // stride + 1
        L = H_out * W_out  # total number of patches per image

        # Create a grid of patch indices (from 0 to L-1)
        patch_indices = torch.arange(L, device=aug_imgs.device)

        # Determine the row (i) and column (j) for each patch in the grid
        grid_i = patch_indices // W_out  # row index for each patch
        grid_j = patch_indices % W_out  # column index for each patch

        # Compute the center position in padded image coordinates:
        # For each patch, its top-left corner in the padded image is at (i*stride, j*stride)
        # Adding (kernel_size//2, kernel_size//2) gives the center.
        center_i_padded = grid_i * stride + (kernel_size // 2)
        center_j_padded = grid_j * stride + (kernel_size // 2)

        # Convert padded coordinates back to original image coordinates by subtracting the padding
        center_i_orig = center_i_padded - padding
        center_j_orig = center_j_padded - padding

        # Stack the row and column indices into a tensor of shape (L, 2)
        patch_centres = torch.stack((center_i_orig, center_j_orig), dim=1)

        assert(patch_centres.shape[0] == tiles.shape[0])

        rays = self.cameras.generate_rays(camera_indices=index, coords = patch_centres)

        text_tokenized = self.model.tokenizer("farm").to(self.device)
        text_embedding = self.model.model.encode_text(text_tokenized)

        embeddings_text = torch.empty((rays.size, text_embedding.shape[1]), device=self.device)

        #origins = []
        #directions = []


        for i in range(rays.size):
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

            origin = rays.origins[i] / self.dataparser_scale

            #calculate ray direction
            fx = self.cameras[0].fx.item()
            fy = self.cameras[0].fy.item()
            cx = self.cameras[0].cx.item()
            cy = self.cameras[0].cy.item()

            #generate intrinsic matrix
            K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]])
            pixel_homog = np.array([patch_centres[i][0], patch_centres[i][1], 1.0])
            direction = np.matmul(np.linalg.inv(K), pixel_homog)
            #normalise to unit vector
            #d_normalized = direction / np.linalg.norm(direction)
            d_normalized = torch.from_numpy(direction)

            ray_orig_homg = torch.cat([origin, torch.tensor([1.0])], dim=0)
            ray_direction_homg = torch.cat([d_normalized, torch.tensor([1.0])], dim=0)

            transformed_origin = np.matmul(inv_transform, ray_orig_homg)
            #transformed_direction = np.matmul(inv_transform, ray_direction_homg)
            transformed_direction = ray_direction_homg

            #convert from homogenous to non-homogenous
            transformed_origin = transformed_origin[:3] / transformed_origin[3]
            transformed_direction = transformed_direction[:3] / transformed_direction[3]

            # convert from opengl to opencv convention
            transformed_origin = torch.Tensor([transformed_origin[1], transformed_origin[0], -transformed_origin[2]])
            #transformed_direction = torch.Tensor([transformed_direction[1], transformed_direction[0], -transformed_direction[2]])


            #origins.append(transformed_origin)
            #directions.append(transformed_direction)

            hit_info = self.SceneGraph.ray_bb_intersection(transformed_origin, transformed_direction)
            if (hit_info['hit']):
                index = hit_info["index"]
                embedding = self.SceneGraph.graph_embeddings[index]
                embeddings_text[i] = embedding
            else:
                embeddings_text[i] = text_embedding

        # origins = torch.cat(origins).view(-1, 3)
        # directions = torch.cat(directions).view(-1, 3)
        #
        # lines = torch.empty((origins.shape[0] * 2, 3))
        # lines[0::2] = origins
        # lines[1::2] = origins + directions
        #
        # fig = go.Figure(  # type: ignore
        #     data=go.Scatter3d(  # type: ignore
        #         x=lines[:, 0],
        #         y=lines[:, 1],
        #         z=lines[:, 2],
        #         marker=dict(
        #             size=4,
        #         ),
        #         line=dict(color="lightblue", width=1),
        #     )
        # )
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
            clip_embeds = (self.model.encode_image(tiles) + embeddings_text) / 2

        clip_embeds /= clip_embeds.norm(dim=-1, keepdim=True)

        clip_embeds = clip_embeds.reshape((self.center_x.shape[0], self.center_y.shape[0], -1))
        clip_embeds = torch.concat((clip_embeds, clip_embeds[:, [-1], :]), dim=1)
        clip_embeds = torch.concat((clip_embeds, clip_embeds[[-1], :, :]), dim=0)
        return clip_embeds.detach().cpu().numpy()
