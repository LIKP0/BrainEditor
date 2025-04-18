# BrainEditor

BrainEditor: Structure-Disentangled Brain MRI Synthesis via Natural Language Prompted Diffusion Image Editing.

Published on ISBI 2025 (oral). [Paper Link](https://github.com/LIKP0/My-tiny-research/tree/main/BrainEditor).

Jialin Li*, Dongwei*, et al. (*denotes equal contribution) 

SUSTech

**Highly Recommend: Have a look at [my research experience for BrainEditor](https://github.com/LIKP0/My-tiny-research/tree/main/BrainEditor)!**

# 1 Install

Part of the code is based on the repositry [instruct-pix2pix](https://github.com/timothybrooks/instruct-pix2pix), which is based on the original [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion).

1. Set up a conda environment by environment.yaml
2. Download the pretrained Stable-Diffusion v1.5 model by Tools/download_pretrained_sd.sh (It also can be found on HuggingFace)

```
conda env create -f environment.yaml
bash Tools/download_pretrained_sd.sh
```
Note: I think it's better to download the original repositry of stable-diffusion (SD) and then add my files. In this way, you can learn SD better.
