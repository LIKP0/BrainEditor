from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from glob import glob

import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset


class BrainDataset(Dataset):
    def __init__(
            self,
            path: str,
    ):
        self.img_list = glob(path + '*')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, i: int):
        name = self.img_list[i]
        image = Image.open(name)
        image = 2 * torch.tensor(np.array(image)).float() / 255 - 1
        example = {'image': image}

        return example
