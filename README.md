# BrainEditor

BrainEditor: Structure-Disentangled Brain MRI Synthesis via Natural Language Prompted Diffusion Image Editing.

Published on ISBI 2025 (oral). [Paper Link](https://github.com/LIKP0/My-tiny-research/tree/main/BrainEditor).

Jialin Li*, Dongwei*, et al. (*denotes equal contribution) 

SUSTech

**Highly Recommend: Take a look at [my research experience for BrainEditor](https://github.com/LIKP0/My-tiny-research/tree/main/BrainEditor)!**

# 1 Install

Part of the code is based on the repositry [instruct-pix2pix](https://github.com/timothybrooks/instruct-pix2pix), which is based on the original [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion).

1. Set up a conda environment.
2. Download the pretrained Stable-Diffusion v1.5 model. (It also can be found on HuggingFace)
3. It's highly recommended to install wandb and [register online](https://wandb.ai/site/) for better training logging.

```
conda env create -f environment.yaml
conda activate BrainEditor
bash Tools/download_pretrained_sd.sh
conda install wandb
```
Note: I think it's better to download the original repositry of stable-diffusion (SD) and then add my files in a right place. In this way, you can learn SD better.

# 2 Generate the Dataset

In this work, we use the public dataset [OASIS-3](https://sites.wustl.edu/oasisbrains/). However, it doesn't include the segmentation mask, which is necessary for following manual dataset generation. We segment the images by FreeSurfer and it really takes a long time. You can concat me for the segmentation data.

1. Generate the "structure-disentangled" text-image dataset.
2. 

```
python create_dataset/create_dataset.py
```

# 3 Train

We pretrain the autoencoder part of SD to adapt for the medical image domain. Although this will result in the loss of generalization ability, experiments shows it can improve the final image quality greatly (2dB on PSNR).

To pretrain the autoencoder

Remember to edit the yaml file according to your configuration:
1. Edit your ckpt_path
2. 

```
python main.py --name BrainEditor_v1 --base configs/train.yaml --train --gpus 2,3,4
```

# 4 Evaluate

