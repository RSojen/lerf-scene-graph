import json

import numpy as np
import torch
from lerf.data.utils.feature_dataloader import FeatureDataloader
from lerf.encoders.image_encoder import BaseImageEncoder
from tqdm import tqdm

from lerf.data.utils.embeddings_directory import Scene_Graph_Nerf_Module
from lerf.encoders.openclip_encoder import (OpenCLIPNetwork,
                                        OpenCLIPNetworkConfig)
from nerfstudio.cameras.cameras import Cameras



class PatchEmbeddingDataloader(FeatureDataloader):
    def __init__(
        self,
        cfg: dict,
        device: torch.device,
        model: BaseImageEncoder,
        image_list: torch.Tensor = None,
        cameras: Cameras = None,
        cache_path: str = None,
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
        # set filepaths
        self.rgb_path = '/home/ritvik/Downloads/Archive 1/kf_image_set_0.monolithic'
        self.depth_path = '/home/ritvik/Downloads/Archive 1/kf_laser_depth_set_0.monolithic'
        self.transforms_path = '/home/ritvik/Downloads/Archive 1/laser_mac_transform.monolithic'
        self.laser_path = '/home/ritvik/Downloads/Archive 1/laser.monolithic'
        self.marker_path = '/home/ritvik/Downloads/Archive 1/farm_markers.monolithic'
        network = OpenCLIPNetworkConfig(
            clip_model_type="ViT-B-16", clip_model_pretrained="laion2b_s34b_b88k", clip_n_dims=512
        )

        self.SceneGraph = None

        self.cameras = cameras

        # instantiate model
        self.model = OpenCLIPNetwork(network)
        print('instantiated model')

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

        self.SceneGraph = Scene_Graph_Nerf_Module(self.rgb_path, self.depth_path, self.transforms_path, self.laser_path, self.marker_path, self.model)

        unfold_func = torch.nn.Unfold(
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        ).to(self.device)

        img_embeds = []
        index = 0
        for img in tqdm(image_list, desc="Embedding images", leave=False):
            img_embeds.append(self._embed_clip_tiles(img.unsqueeze(0), unfold_func, index))
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

        # Parameters from your patch extraction:
        kernel_size = self.kernel_size  # e.g., 16
        stride = self.stride  # non-overlapping patches

        # Original image dimensions (H, W) should be known or available from the camera.
        H, W = aug_imgs.shape[-2:]  # assuming aug_imgs is [B, C, H, W]

        # Compute the top-left indices for each patch.
        rows = torch.arange(0, H - kernel_size + 1, stride)
        cols = torch.arange(0, W - kernel_size + 1, stride)
        grid_cols, grid_rows = torch.meshgrid(cols, rows, indexing='xy')  # shape: [num_patches_x, num_patches_y]

        # Compute patch centre coordinates (in pixel indices).
        patch_centres = torch.stack(
            [grid_rows + kernel_size // 2, grid_cols + kernel_size // 2], dim=-1
        )  # shape: [num_patches_x, num_patches_y, 2]

        # Flatten to a list of coordinates.
        patch_centres = patch_centres.reshape(-1, 2)  # each row is [row, col]

        rays = self.cameras.generate_rays(camera_index=index, indices = patch_centres)
        for ray in rays:
            hit_info = self.SceneGraph.ray_bb_intersection(ray.origin, ray.direction)
            index = hit_info["index"]


        with torch.no_grad():
            clip_embeds = self.model.encode_image(tiles)
            clip_embeds_text = self.model.encode_text()
        clip_embeds /= clip_embeds.norm(dim=-1, keepdim=True)

        clip_embeds = clip_embeds.reshape((self.center_x.shape[0], self.center_y.shape[0], -1))
        clip_embeds = torch.concat((clip_embeds, clip_embeds[:, [-1], :]), dim=1)
        clip_embeds = torch.concat((clip_embeds, clip_embeds[[-1], :, :]), dim=0)
        return clip_embeds.detach().cpu().numpy()
