# Installation and setup
lerf-scene-graph requires a modified version of nerstudio which can be installed as follows


### Prerequisites

You must have an NVIDIA video card with CUDA installed on the system. This library has been tested with version 11.8 of CUDA. You can find more information about installing CUDA [here](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)

### Create environment

Nerfstudio requires `python >= 3.8`. We recommend using conda to manage dependencies. Make sure to install [Conda](https://docs.conda.io/miniconda.html) before proceeding.

```bash
conda create --name nerfstudio -y python=3.8
conda activate nerfstudio
pip install --upgrade pip
```

### Dependencies

Install PyTorch with CUDA (this repo has been tested with CUDA 11.7 and CUDA 11.8) and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn).
`cuda-toolkit` is required for building `tiny-cuda-nn`.

For CUDA 11.8:

```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

See [Dependencies](https://github.com/nerfstudio-project/nerfstudio/blob/main/docs/quickstart/installation.md#dependencies)
in the Installation documentation for more.

### Installing nerfstudio

```bash
git clone https://github.com/RSojen/nerf-scene-graph.git
cd nerfstudio
pip install --upgrade pip setuptools
pip install -e .
```

# Integration with scene graphs
Once the markers.json file has been generated:
### 0. Replace the marker_path variable
replace the variable marker_path in lerf/data/utils/pyramid_embedding_dataloader with the path to the json file from above. 
### 1. Lerf installation
cd to this folder then run `python -m pip install -e .`
### 2. Updating the command line interface
run `ns-install-cli`
### 3. train the network
Then once lerf has been added to nerfstudio using the above instructions, the network can be trained given a path to a transforms.json file as follows: `ns-train lerf --data {PATH TO TRANSFORMS.JSON}`.

### 4. Generating relevany maps for cameras
Once the network has been trained, the script lerf_render_image.py in lerf/scripts can be run with the following parameters int the python file. the config_dir variable should be set to wherever nerfstudio saved the model checkpoints after the training process
and the positives (text query) can be set using the set_positives function. By default the camera poses used are from the training dataset

## Bibtex
If you find this useful, please cite the paper!
<pre id="codecell0">@inproceedings{lerf2023,
&nbsp;author = {Kerr, Justin and Kim, Chung Min and Goldberg, Ken and Kanazawa, Angjoo and Tancik, Matthew},
&nbsp;title = {LERF: Language Embedded Radiance Fields},
&nbsp;booktitle = {International Conference on Computer Vision (ICCV)},
&nbsp;year = {2023},
} </pre>
