# LatentFusion

 * [[project page](https://keunhong.com/publications/latentfusion/)]
 * [[paper](https://arxiv.org/abs/1912.00416)]

## Citing LatentFusion
If you find the LatentFusion code or data useful, please consider citing:

```bibtex
@inproceedings{park2019latentfusion,
  title={LatentFusion: End-to-End Differentiable Reconstruction and Rendering for Unseen Object Pose Estimation},
  author={Park, Keunhong and Mousavian, Arsalan and Xiang, Yu and Fox, Dieter},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```

## Setup

Please start by installing [Miniconda3](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html) with Python 3.7 or above.

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh

Then create a Conda environment with our environment file:

    conda env create -n latentfusion -f environment.yml
    conda activate latentfusion
    
Please make sure the project root is added to the `$PYTHONPATH`. We provide a simple script for this:

    # Activates the Conda environment and sets PYTHONPATH.
    source env.sh
    
For training, we make use of Automatic Mixed Precision. Until PyTorch 1.6 is released, you must install the nightly version of PyTorch.

    conda install pytorch torchvision cudatoolkit=10.2 -c pytorch-nightly

We've exluded PyTorch from the `environment.yml` file for this reason.


## Trained Model

We provide a trained model. You can download the weights from [here](https://drive.google.com/file/d/18NIGeCO4U5fgTpqH3bBTXfk1Ib1f0Seh/view). The weights are licensed under a Creative Commons license. Please see the [weights license](WEIGHTS_LICENSE) for details.

You can use the model like this:
```python
import torch
from latentfusion.recon.inference import LatentFusionModel
checkpoint = torch.load('path-to-checkpoint')
model = LatentFusionModel.from_checkpoint(checkpoint)
```

Please see the [example notebook](examples/pose_estimation.ipynb) for details.


## Dataset Download

### BOP/LINEMOD

We use the BOP version of LINEMOD and other datasets. You can get them at the [BOP website](https://bop.felk.cvut.cz/home/).

### MOPED

You can download the MOPED dataset at the [project page](https://keunhong.com/publications/latentfusion/).


## Pose Estimation

We provide an example script for pose estimation in the `examples` directory in the form of a Jupyter notebook. Download the weights from [here](https://drive.google.com/file/d/18NIGeCO4U5fgTpqH3bBTXfk1Ib1f0Seh/view)
and open the notebook with Jupyter.


## Training

To train LatentFusion, we recommend that you use at least 4 RTX 2080 Ti GPUs. First, modify `tools/train/train.sh`
with the correct paths to ShapeNet and MS-COCO.

You must first pre-process ShapeNet with our preprocessing script in `tools/dataset/preprocess_shapenet.py`.
This requires that you have [Blender](https://www.blender.org/) installed.

    sudo apt install blender
    blender -P tools/dataset/preprocess_shapenet.py -- "$SHAPENET_PATH" "$OUT_PATH" --strip-materials --out-name ShapeNetCore-nomat
    
Once you have all of the data in place simply call the training script:

    bash tools/train/train.sh

Training will roughly take 2 weeks on 4x RTX 2080 Ti GPUs and 1 week on 4x Tesla V100 GPUs.


