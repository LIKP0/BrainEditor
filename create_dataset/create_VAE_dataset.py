import os
from glob import glob
import nibabel as nib
import numpy as np
import random
from PIL import Image
import imageio
from brain_utils import extract_img_slice, pad_to_square


# Create the dataset for nnUNet segmentation evaluation


def extract_label_slice(data, save_name):
    # cls = {2: 255, 41: 255, 3: 192, 42: 192, 4: 150, 43: 150, 17: 75, 53: 75} # for visualization
    cls = {2: 1, 41: 1, 3: 2, 42: 2, 4: 3, 43: 3, 17: 4, 53: 4}  # for training
    img = np.zeros(data.shape)
    for i in cls.keys():
        img[np.where(data == i)] = cls[i]
    img = np.uint8(img)
    img = Image.fromarray(img)
    # img = img.convert('RGB')
    img.save(save_name, 'png')
    print(f'Extract to {save_name}')


if __name__ == '__main__':
    img_src = '/data1/lijl/OASIS2/image/'
    label_src = '/data1/lijl/OASIS2/label/'
    dst_dir = '/data1/lijl/latent-diffusion/dataset/Dataset099/train/'

    X = 80  # 160
    Y = [i for i in range(70, 101, 2)]  # 192
    Z = 112  # 224
    padding = 192

    img_list = sorted(glob(img_src + '*.nii.gz'))
    print('All images num: ', len(img_list))
    for img_name in img_list:
        index = img_name.split('/')[-1].split('.')[0]
        print(f'Processing {index}...')

        img = nib.load(img_name).get_fdata()
        for s in Y:  # Extract Slices of img and seg in Y
            print(f'At Slice {s}')
            img1s = img[:, s, 12:204]  # 160, 192
            img1s = pad_to_square(img1s, padding)  # 192, 192
            img1 = Image.fromarray(img1s).resize((256, 256), Image.Resampling.BILINEAR).convert('RGB')  # resize will make float
            img1 = np.array(img1, dtype=np.uint8)
            Image.fromarray(img1).save(os.path.join(dst_dir, f'{index}_XZ{s}_0000.jpg'))

