from pathlib import Path
import cv2
import torch
import numpy as np

from nerfstudio.cameras.rays import RayBundle

from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.cameras import Cameras, CameraType, RayBundle

from lerf.lerf_pipeline import LERFPipeline


class LerfRenderer():
    def __init__(self,
                 config = None,
                 device = None):
        self.device = device
        self.config = config
        _, self.pipeline, _, _ = eval_setup(Path(config))
        

    def render_image(self, camera, dir, index):
        camera = camera.to(self.device)
        torch.cuda.empty_cache()
        outputs = self.pipeline.model.get_outputs_for_camera(camera)
        print(outputs.keys())
        relevancy = outputs['composited_0'].cpu().detach().numpy()
        rgb = outputs['rgb'].cpu().detach().numpy()

        print(rgb)

        rgb = (rgb * 255).astype(np.uint8)
        relevancy = (relevancy * 255).astype(np.uint8) 

        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        #write image to output directory
        out_file = dir + '/render_' + str(index) + '.png'
        out_rgb = dir + '/rgb_' + str(index) + '.png'
        cv2.imwrite(out_file, relevancy)
        cv2.imwrite(out_rgb, rgb)

        
    def set_positives(self, positives):
        self.pipeline.model.image_encoder.set_positives(positives)

    def render_training_data(self, out_dir):
        cameras = self.pipeline.datamanager.train_dataset.cameras
        for i in range(len(cameras)):
            self.render_image(cameras[i], out_dir, i)

        



if __name__ == "__main__":
    config_dir = '/home/paperspace/code/lerf-scene-graph/outputs/outputs/lerf/2025-03-04_141604/config.yml'
    device = torch.device('cuda')
    renderer = LerfRenderer(config=config_dir, device=device)
    renderer.set_positives(['all plants in the first section'])
    renderer.render_training_data('/home/paperspace/code/lerf-scene-graph/lerf/output_renders')