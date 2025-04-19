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
2. Generate the image-label dataset to train the nnUNet, which is used for structure change accuracy evaluation.
3. Generate the image-image dataset to train the autoencoder of SD.

```
python create_dataset/create_dataset.py
python create_dataset/create_nnUnet_dataset.py
python create_dataset/create_VAE_dataset.py
```

# 3 Train

**You need at least 40GB GPU for a reasonable training time!**

First, we pretrain the autoencoder part of SD to adapt for the medical image domain. Although this will result in the loss of generalization ability, experiments shows it can improve the final image quality greatly (2dB on PSNR).

To pretrain the autoencoder (very easy!):
1. Clone the original [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion) repositry.
2. Place the files in AutoEncoder_pretrain folder to the right place.
3. Train in the ldm directory with our own medical images data.

```
python main.py --base AutoEncoder_pretrain/Brain_VAE.yaml --train --name brain_vae --gpus 3,4
```

Then, we train the image editing model of BrainEditor. Remember to edit the yaml file according to your configuration:
1. Edit your ckpt_path, including the pretrained SD and autoencoder checkpoint path.
2. Edit your data path

```
python main.py --base configs/train.yaml --train --name BrainEditor_v1 --gpus 3,4
```

# 4 Evaluate



