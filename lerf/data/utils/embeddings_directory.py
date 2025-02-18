import torch

from lerf.encoders.image_encoder import BaseImageEncoder
from lerf.encoders.openclip_encoder import (OpenCLIPNetwork,
                                        OpenCLIPNetworkConfig)
import sys
import os

import numpy as np
# Determine the build directory
build_dir = '/home/ritvik/aru_sil_core/interfaces/build/temp.linux-x86_64-cpython-38'
module_path = None
for root, dirs, files in os.walk(build_dir):
    for file in files:
        print(file)
        if file.startswith('aru_state_estimator') and (file.endswith('.so') or file.endswith('.pyd')):
            module_path = root
            break

# Add the build directory to the system path
sys.path.append(module_path)
#add build files to path
sys.path.insert(0,"/home/ritvik/aru_sil_core/interfaces/build/temp.linux-x86_64-cpython-38/lib")
import aru_recon_interface


class Scene_Graph_Nerf_Module():
    def __init__(
            self,
            rgb_filepath: str = None,
            depth_filepath: str = None,
            transform_path: str = None,
            laser_path: str = None,
            markers_path: str = None,
            model: BaseImageEncoder = None,
            device: torch.device = None
    ):
        self.recon_interface = aru_recon_interface.ReconInterface(rgb_filepath, depth_filepath, transform_path, laser_path, False)
        print('finished reading monolithics')
        self.markers = self.recon_interface.read_markers(markers_path)
        self.device = device
        self.object_points = self.markers[0]
        #print(self.object_points)
        self.object_colors = self.markers[1]
        #print(self.object_colors)
        self.object_labels = self.markers[2]
        #print(self.object_labels)
        self.num_objects = self.markers[3]
        #print(self.num_objects)
        self.row_labels = self.markers[4]
        #print(self.row_labels)
        self.box_points = self.markers[5]
        #print(self.box_points)
        self.nodes = self.markers[6]
        self.descriptors = self.markers[7]
        print(self.descriptors)

        #build the bvh
        objects = []

        for element, box in enumerate(self.box_points):
            bounding_box={}
            lower_bounds = box[:3]
            upper_bounds = box[3:]
            bounding_box["bounding_box"] = (lower_bounds, upper_bounds)
            bounding_box["centroid"] = (lower_bounds + upper_bounds)/2
            bounding_box["index"] = element
            objects.append(bounding_box)

        self.bvh_root = self.build_bvh(objects)

        self.graph_embeddings = self._calculate_embeddings(model)

    def _calculate_embeddings(self, model: BaseImageEncoder):
        # calculate CLIP text embedding for a hierachical text description
        graph_embeddings = {}

        for i in range(len(self.box_points)):
            index = i

            print("encoding text...")
            print(self.descriptors[i])

            text_tokenized = model.tokenizer(self.descriptors[i]).to(self.device)
            print(text_tokenized.device)
            text_embedding = model.model.encode_text(text_tokenized)

            print("finished encoding")
            graph_embeddings[index] = text_embedding

        return graph_embeddings

    def build_bvh(self, objects):
        if len(objects) == 1:
            # Leaf node
            return BVHNode(objects[0]['bounding_box'], leaf=True, index=objects[0]['index'])

        # Compute the bounding box for all objects
        all_mins = np.min([obj['bounding_box'][0] for obj in objects], axis=0)
        all_maxs = np.max([obj['bounding_box'][1] for obj in objects], axis=0)
        bounding_box = (all_mins, all_maxs)

        # Determine the axis with the largest spread
        extents = all_maxs - all_mins
        split_axis = np.argmax(extents)

        # Sort objects along the split axis
        objects.sort(key=lambda obj: obj['centroid'][split_axis])

        # Split the list into two halves
        mid = len(objects) // 2
        left_objects = objects[:mid]
        right_objects = objects[mid:]

        # Recursively build the BVH
        left_node = self.build_bvh(left_objects)
        right_node = self.build_bvh(right_objects)

        return BVHNode(bounding_box, left=left_node, right=right_node)

    def traverse_bvh(self, node, ray_origin, ray_direction, hit_info=None):
        if hit_info is None:
            hit_info = {'hit': False, 't': float('inf'), 'index': None}

        if node is None:
            return hit_info

        # Check ray intersection with the node's bounding box
        intersects, t = self.ray_intersects_aabb(ray_origin, ray_direction, *node.bounding_box)
        if not intersects or t > hit_info['t']:
            return hit_info

        if node.leaf:
            # Update hit information
            hit_info['hit'] = True
            hit_info['t'] = t
            hit_info['index'] = node.index
            return hit_info

        # Traverse child nodes
        hit_info = self.traverse_bvh(node.left, ray_origin, ray_direction, hit_info)
        hit_info = self.traverse_bvh(node.right, ray_origin, ray_direction, hit_info)

        return hit_info

    def ray_bb_intersection(self, ray_origin, ray_direction):
        hit_info = self.traverse_bvh(self.bvh_root, ray_origin, ray_direction)
        return hit_info

    def ray_intersects_aabb(ray_origin, ray_direction, box_min, box_max):
        tmin = (box_min - ray_origin) / ray_direction
        tmax = (box_max - ray_origin) / ray_direction

        t1 = np.minimum(tmin, tmax)
        t2 = np.maximum(tmin, tmax)

        t_enter = np.max(t1)
        t_exit = np.min(t2)

        if t_enter > t_exit or t_exit < 0:
            return False, None  # No intersection

        return True, t_enter  # Intersection occurs


class BVHNode:
    def __init__(self, bounding_box, left=None, right=None, leaf=False, index=None):
        self.bounding_box = bounding_box  # (min_point, max_point)
        self.left = left
        self.right = right
        self.leaf = leaf
        self.index = index  # Index of the object in the original list (used for leaf nodes)


if __name__ == "__main__":
    network = OpenCLIPNetworkConfig(
        clip_model_type="ViT-B-16", clip_model_pretrained="laion2b_s34b_b88k", clip_n_dims=512
    )

    #instantiate model
    model = OpenCLIPNetwork(network)
    print('instantiated model')

    #set filepaths
    rgb_path = '/home/ritvik/Downloads/Archive 1/kf_image_set_0.monolithic'
    depth_path = '/home/ritvik/Downloads/Archive 1/kf_laser_depth_set_0.monolithic'
    transforms_path = '/home/ritvik/Downloads/Archive 1/laser_mac_transform.monolithic'
    laser_path = '/home/ritvik/Downloads/Archive 1/laser.monolithic'
    marker_path = '/home/ritvik/Downloads/Archive 1/farm_markers.monolithic'

    device = torch.device("cuda")

    module = Scene_Graph_Nerf_Module(rgb_path, depth_path, transforms_path, laser_path, marker_path, model, device)















