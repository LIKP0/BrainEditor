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
    imagesTr = '/data1/lijl/nnUNet/nnUNet_raw/nnUNet_raw/Dataset097_OASIS2_2D/imagesTr/'
    imagesTs = '/data1/lijl/nnUNet/nnUNet_raw/nnUNet_raw/Dataset097_OASIS2_2D/imagesTs/'
    labelsTr = '/data1/lijl/nnUNet/nnUNet_raw/nnUNet_raw/Dataset097_OASIS2_2D/labelsTr/'
    labelsTs = '/data1/lijl/nnUNet/nnUNet_raw/nnUNet_raw/Dataset097_OASIS2_2D/labelsTs/'

    X = 80  # 160
    Y = [60, 70, 80, 90, 100, 110, 120]  # 192
    Z = 112  # 224
    padding = 256

    img_list = glob(img_src + '*.nii.gz')
    random.seed(0)
    random.shuffle(img_list)
    test_split = img_list[:len(img_list) // 10]
    train_and_val_split = img_list[len(img_list) // 10:]
    print('All images num: ', len(img_list))
    print('Train and val num: ', len(train_and_val_split))
    print('Test num: ', len(test_split))

    for img_name in train_and_val_split:
        index = img_name.split('/')[-1].split('.')[0]
        print(f'Processing {index}...')

        seg_name = label_src + index + '_seg.nii.gz'
        img = nib.load(img_name).get_fdata()
        seg = nib.load(seg_name).get_fdata()
        for s in Y:  # Extract Slices of img and seg in Y
            print(f'At Slice {s}')
            img1s = img[:, s, :]
            seg1s = seg[:, s, :]
            img1s = pad_to_square(img1s, padding)
            seg1s = pad_to_square(seg1s, padding)

            extract_img_slice(data=img1s, save_name=os.path.join(imagesTr, f'{index}_XZ{s}_0000.png'))
            extract_label_slice(data=seg1s, save_name=os.path.join(labelsTr, f'{index}_XZ{s}.png'))

    for img_name in test_split:
        index = img_name.split('/')[-1].split('.')[0]
        print(f'Processing {index}...')

        seg_name = label_src + index + '_seg.nii.gz'
        img = nib.load(img_name).get_fdata()
        seg = nib.load(seg_name).get_fdata()
        for s in Y:  # Extract Slices of img and seg in Y
            print(f'At Slice {s}')
            img1s = img[:, s, :]
            seg1s = seg[:, s, :]
            img1s = pad_to_square(img1s, padding)
            seg1s = pad_to_square(seg1s, padding)

            extract_img_slice(data=img1s, save_name=os.path.join(imagesTs, f'{index}_XZ{s}_0000.png'))
            extract_label_slice(data=seg1s, save_name=os.path.join(labelsTs, f'{index}_XZ{s}.png'))
