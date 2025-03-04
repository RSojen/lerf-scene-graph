from typing import Dict, Literal, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import HashEncoding, NeRFEncoding, SHEncoding
from nerfstudio.field_components.field_heads import (
    FieldHeadNames,
    PredNormalsFieldHead,
    SemanticFieldHead,
    TransientDensityFieldHead,
    TransientRGBFieldHead,
    UncertaintyFieldHead,
)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, get_normalized_directions

class Marine_Nerf_Field(Field):

    aabb: Tensor

    def __init__(
        self,
        aabb: Tensor,
        num_images: int,
        num_layers=2,
        hidden_dim: int = 128,
        geo_feat_dim: int = 15,
        num_levels: int = 16,
        base_res: int = 16,
        max_res: int = 4096,
        log2_hashmap_size: int = 21,
        num_layers_color: int = 4,
        num_layers_transient: int = 2,
        features_per_level: int = 2,
        hidden_dim_color: int = 128,
        hidden_dim_transient: int = 64,
        appearance_embedding_dim: int = 8,
        transient_embedding_dim: int = 16,
        use_transient_embedding: bool = False,
        use_semantics: bool = False,
        num_semantic_classes: int = 100,
        pass_semantic_gradients: bool = False,
        use_pred_normals: bool = False,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
        implementation: Literal["tcnn", "torch"] = "tcnn",
        ) -> None:

        super().__init__()

        self.register_buffer("aabb", aabb)
        self.geo_feat_dim = geo_feat_dim

        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))

        self.spatial_distortion = spatial_distortion
        self.num_images = num_images
        self.appearance_embedding_dim = appearance_embedding_dim
        self.embedding_appearance = Embedding(self.num_images, self.appearance_embedding_dim)
        self.use_average_appearance_embedding = use_average_appearance_embedding
        self.use_transient_embedding = use_transient_embedding
        self.use_semantics = use_semantics
        self.use_pred_normals = use_pred_normals
        self.pass_semantic_gradients = pass_semantic_gradients
        self.base_res = base_res

        #specify method used for direction encoding
        self.direction_encoding = SHEncoding(
            levels = 4,
            implementation = implementation
        )

        #specify method used for position encoding
        self.position_encoding = NeRFEncoding(
            in_dim=2,
            num_frequencies = 2,
            min_freq_exp=0,
            max_freq_exp=2 - 1,
            implementation=implementation
        )

        self.mlp_base_grid = HashEncoding(
            num_levels=num_levels,
            min_res=base_res,
            max_res=max_res,
            log2_hashmap_size=log2_hashmap_size,
            features_per_level=features_per_level,
            implementation=implementation,
        )

        self.mlp_base_mlp = MLP(
            in_dim=self.mlp_base_grid.get_out_dim(),
            num_layers=num_layers,
            layer_width=hidden_dim,
            out_dim=1 + self.geo_feat_dim,
            activation=nn.ReLU(),
            out_activation=None,
            implementation=implementation,
        )

        self.mlp_base = torch.nn.Sequential(self.mlp_base_grid,self.mlp_base_mlp)

        #create sky network to predict color for pixels that pass through the sky
        self.mlp_skynet = MLP(
            in_dim=self.direction_encoding.get_out_dim(),
            num_layers=num_layers_color,
            layer_width=hidden_dim_color,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation=implementation
        )

        #create normal color network for pixels that don't pass through sky regions
        self.mlp_head = MLP(
            in_dim=self.direction_encoding.get_out_dim() + self.geo_feat_dim,
            num_layers=num_layers_color,
            layer_width=hidden_dim_color,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation=implementation
        )

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        """Computes densities and returns them"""
        if self.spatial_distortion is not None:
            positions=ray_samples.frustums.get_positions()
            #apply relevant spatial distortion
            positions=self.spatial_distortion(positions)
            positions=(positions+2.0)/4.0
        else:
            positions=SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)

        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True
        positions_flat = positions.view(-1, 3)
        h = self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        self._density_before_activation = density_before_activation

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation.to(positions))
        density = density * selector[..., None]


        return density, base_mlp_out

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[Tensor]
    ) -> Dict[FieldHeadNames, Tensor]:

        assert density_embedding is not None
        outputs = {}
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        camera_indices = ray_samples.camera_indices.squeeze()

        directions = get_normalized_directions(ray_samples.frustums.directions)
        ray_directions_only = directions[:,0,:].squeeze()

        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)
        d_sky = self.direction_encoding(ray_directions_only)

        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        # appearance
        if self.training:
            embedded_appearance = self.embedding_appearance(camera_indices)
        else:
            if self.use_average_appearance_embedding:
                embedded_appearance = torch.ones(
                    (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                ) * self.embedding_appearance.mean(dim=0)
            else:
                embedded_appearance = torch.zeros(
                    (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                )
        embedded_appearance_sky = embedded_appearance[:,0,...]

        h = torch.cat(
            [
                d,
                density_embedding.view(-1, self.geo_feat_dim)
            ],
            #+ ([embedded_appearance.view(-1, self.appearance_embedding_dim)] if embedded_appearance is not None else []),
            dim=-1
        )

        h_sky = d_sky

        rgb_sky = self.mlp_skynet(h_sky).to(directions)

        rgb_normal = self.mlp_head(h).view(*outputs_shape, -1).to(directions)

        outputs.update({FieldHeadNames.RGB: rgb_normal, "rgb_sky": rgb_sky})

        return outputs
















