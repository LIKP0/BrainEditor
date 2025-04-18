"""
1. real image + two sentences
2. real image + one sentence
3. disentangled image (crop & paste) + one sentence
4. disentangled image (possion edit) + one sentence
This is the fourth step of the study: possion + one
"""

import os
from glob import glob
import nibabel as nib
import numpy as np
import json
import cv2
import random
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

VE_interval = [0, 10, 20, 100]
BM_interval = [0, 25]

""" For nnUNet
"labels": {
        "background": 0,
        "white Matter": 1,
        "grey matter": 2,
        "ventricle": 3,
        "hippocampus": 4
    },
"""

def pad_to_square(data, padding):
    rows, cols = data.shape
    row_pad = (padding - rows) // 2
    col_pad = (padding - cols) // 2
    data = np.pad(data, ((row_pad, row_pad), (col_pad, col_pad)), 'constant')
    data = data.T[::-1]  # counter-clockwise rotate 90 degree
    return data


def find_interval(interval, val):
    for i in range(len(interval) - 1):
        if interval[i] <= val < interval[i + 1]:
            return i + 1
    return 0


def ventricle_prompt(seg1s, seg2s):
    # Volume change ratio (B-A)/A*100:
    Va = np.sum(seg1s == 3)  # Ventricle
    Vb = np.sum(seg2s == 3)

    smooth = 1e-5
    # ratio = math.floor((Vb - Va) / (Va + smooth) * 100)
    ratio = (Vb - Va) / (Va + smooth) * 100
    print(f'Ventricle volume change: {ratio}%')
    return find_interval(VE_interval, ratio)
    # return ratio


def cortex_prompt(seg1s, seg2s):
    # Volume change ratio (B-A)/A*100:
    Va = np.sum(seg1s == 1) + np.sum(seg1s == 2)  # WM + GM
    Vb = np.sum(seg2s == 1) + np.sum(seg2s == 2)
    # Va = np.sum(seg1s == 3) + np.sum(seg1s == 42)  # GM
    # Vb = np.sum(seg2s == 3) + np.sum(seg2s == 42)

    smooth = 1e-5
    # ratio = math.floor((Vb - Va) / (Va + smooth) * 100)
    ratio = (Vb - Va) / (Va + smooth) * 100
    print(f'Cortex volume change: {ratio}%')
    return find_interval(BM_interval, -1 * ratio)
    # return ratio


def convert_mask(data):  # to 255 for openCV using
    cls = {3: 255}  # label convert for nnUNet
    img = np.zeros(data.shape, dtype=np.uint8)
    for i in cls.keys():
        img[np.where(data == i)] = cls[i]
    return img


def poisson_edit(before, after, mask):
    w, h = mask.shape
    minx, miny = 1000, 1000
    maxx, maxy = -1, -1
    for i in range(0, w):  # locate the min matrix that contains the whole ROI
        for j in range(0, h):
            if mask[i, j] == 255:
                minx = min(i, minx)
                miny = min(j, miny)
                maxx = max(i, maxx)
                maxy = max(j, maxy)
    if abs(minx - 128) < abs(maxx - 128):  # make the ROI matrix symmetry by x=128
        minx = 128 - abs(maxx - 128)
    else:
        maxx = 128 + abs(minx - 128)
    if abs(miny - 128) < abs(maxy - 128):
        miny = 128 - abs(maxy - 128)
    else:
        maxy = 128 + abs(miny - 128)

    mask_matrix = np.zeros(mask.shape, dtype='uint8')
    biasx, biasy = 10, 10  # add bias to cover the ROI better
    mask_matrix[minx - biasx:maxx + biasx, miny - biasy:maxy + biasy] = 255
    # center = ((maxx + minx) // 2, (maxy + miny) // 2)
    center = (128, 128)
    # mask_matrix = np.zeros(mask.shape, dtype='uint8')
    # mask_matrix[100:200, 100:200] = 255
    print(minx, miny, maxx, maxy, center)

    # clone after's larger ventricle to before ==> before's cortex + after's ventricle ==> VX
    try:
        VX = cv2.seamlessClone(src=after, dst=before, mask=mask_matrix, p=center, flags=cv2.NORMAL_CLONE)
    except Exception as e:
        print(e)
        VX = after
    return VX


# Use label with nnUNet pred
if __name__ == '__main__':
    img_src = '/data1/lijl/OASIS2/image/'
    label_src = '/data1/lijl/instruct-pix2pix/data/OASIS2_dataset_v8_pre2/nnUNet_pred_label/'
    train_dataset_dir = "/data1/lijl/instruct-pix2pix/data/OASIS2_dataset_v11_stage1/"
    test_dataset_dir = "/data1/lijl/instruct-pix2pix/data/V11_stage1_testbench/testdata/"
    X = 80  # 160
    Y = [i for i in range(70, 101, 2)]  # 192
    Z = 112  # 224
    padding = 192

    if not os.path.exists(train_dataset_dir):
        os.makedirs(train_dataset_dir)
    if not os.path.exists(test_dataset_dir):
        os.makedirs(test_dataset_dir)

    img_list = sorted(glob(img_src + '*.nii.gz'))
    img_dict = {}
    for img in img_list:
        subject = img.split('/')[-1][:9]
        if subject not in img_dict:
            img_dict[subject] = []
        img_dict[subject].append(img)

    total_num = 0
    omitted_img = []
    VE_prompt = {}
    BM_prompt = {}
    seeds = []
    random.seed(0)
    for subject in img_dict:
        print(f'##### Processing {subject} #####')

        subseed_arr = []
        images = img_dict[subject]
        for m in range(len(images) - 1):
            img1n = images[m]
            index1 = img1n.split('/')[-1].split('.')[0]
            img1 = np.uint32(nib.load(img1n).get_fdata())
            for n in range(m + 1, len(images)):
                img2 = images[n]
                index2 = img2.split('/')[-1].split('.')[0]
                presubseed = f'{subject}_{img1n[-10:-7]}{img2[-10:-7]}'  # OAS2_0004_MR1MR2, OAS2_0004_MR1MR3...
                print(presubseed)

                img2 = np.uint32(nib.load(img2).get_fdata())
                for s in Y:  # Extract Slices of img and seg in Y
                    total_num += 1
                    seg1s = os.path.join(label_src, f'{index1}_XZ{s}.png')
                    seg2s = os.path.join(label_src, f'{index2}_XZ{s}.png')
                    print(f'At Slice {s}')
                    print('Use seg1: ', seg1s)
                    print('Use seg2: ', seg2s)

                    origin = pad_to_square(img1[:, s, 12:204], padding)  # 224, 160 --> 192, 192
                    target = pad_to_square(img2[:, s, 12:204], padding)

                    orig = np.array(Image.fromarray(origin).convert('L'))
                    targ = np.array(Image.fromarray(target).convert('L'))
                    psnr = compare_psnr(orig[:, 20:172], targ[:, 20:172])
                    print(f'PSNR: {psnr}')
                    if psnr < 25:
                        omitted_img.append(f'{presubseed}_s{s}')
                        print('Omitted for low PSNR.')
                        continue

                    # resize will make float
                    origin = Image.fromarray(origin).resize((256, 256), Image.Resampling.BILINEAR).convert('RGB')
                    target = Image.fromarray(target).resize((256, 256), Image.Resampling.BILINEAR).convert('RGB')
                    seg1s = np.array(Image.open(seg1s).convert('L'))
                    seg2s = np.array(Image.open(seg2s).convert('L'))

                    if random.random() < 0.05:
                        dataset_dir = test_dataset_dir
                        train = False
                    else:
                        dataset_dir = train_dataset_dir
                        train = True

                    # Generate text prompt through labels
                    VE_index = ventricle_prompt(seg1s, seg2s)
                    BM_index = cortex_prompt(seg1s, seg2s)
                    if VE_index != 0:
                        prompt = {
                            'input': '',  # no need for training
                            'edit': f'Change the ventricle to V{VE_index}',
                            'output': '',
                            'url': ''
                        }

                        seed = f'{presubseed}_XZ{s}_V{VE_index}'  # one slice one seed one prompt
                        seed_path = os.path.join(dataset_dir, seed)
                        if not os.path.exists(seed_path):
                            os.mkdir(seed_path)
                        with open(os.path.join(seed_path, 'prompt.json'), 'w') as f:
                            f.write(json.dumps(prompt))

                        # use target's ventricle replace the origin on the same position
                        # use poisson image editing

                        target_ve = poisson_edit(np.array(origin), np.array(target), convert_mask(seg2s))
                        origin.save(os.path.join(seed_path, '0_0.jpg'), format='PNG')  # before edit
                        Image.fromarray(target_ve).save(os.path.join(seed_path, '0_1.jpg'), format='PNG')  # after edit
                        if train:
                            seeds.append([seed, ['0']])  # OAS2_0004_MR1MR2_XZ60_V1/0, OAS2_0004_MR1MR2_XZ60_V1/1

                    if BM_index != 0:
                        prompt = {
                            'input': '',
                            'edit': f'Change the cortex to C{BM_index}',
                            'output': '',
                            'url': ''
                        }

                        seed = f'{presubseed}_XZ{s}_C{BM_index}'  # one slice one seed one prompt
                        seed_path = os.path.join(dataset_dir, seed)
                        if not os.path.exists(seed_path):
                            os.mkdir(seed_path)
                        with open(os.path.join(seed_path, 'prompt.json'), 'w') as f:
                            f.write(json.dumps(prompt))

                        # use origin's ventricle replace the target on the same position
                        target_c = poisson_edit(np.array(target), np.array(origin), convert_mask(seg2s))
                        origin.save(os.path.join(seed_path, '0_0.jpg'), format='PNG')  # before edit
                        Image.fromarray(target_c).save(os.path.join(seed_path, '0_1.jpg'), format='PNG')  # after edit
                        if train:
                            seeds.append([seed, ['0']])  # OAS2_0004_MR1MR2_XZ60_C1/0, OAS2_0004_MR1MR2_XZ60_C1/1

                    if VE_index != 0 and BM_index != 0:
                        prompt = {
                            'input': '',
                            'edit': f'Change the ventricle to V{VE_index}, Change the cortex to C{BM_index}',
                            'output': '',
                            'url': ''
                        }

                        seed = f'{presubseed}_XZ{s}_V{VE_index}C{BM_index}'  # one slice one seed one prompt
                        seed_path = os.path.join(dataset_dir, seed)
                        if not os.path.exists(seed_path):
                            os.mkdir(seed_path)
                        with open(os.path.join(seed_path, 'prompt.json'), 'w') as f:
                            f.write(json.dumps(prompt))

                        origin.save(os.path.join(seed_path, '0_0.jpg'), format='PNG')  # before edit
                        target.save(os.path.join(seed_path, '0_1.jpg'), format='PNG')  # after edit
                        if train:
                            seeds.append([seed, ['0']])

                    if VE_index in VE_prompt.keys():
                        VE_prompt[VE_index] += 1
                    else:
                        VE_prompt[VE_index] = 1
                    if BM_index in BM_prompt.keys():
                        BM_prompt[BM_index] += 1
                    else:
                        BM_prompt[BM_index] = 1

                    print('############################')

    with open(os.path.join(train_dataset_dir, 'seeds.json'), 'w') as f:
        f.write(json.dumps(seeds))
    print('All subjects finished.')

    print('VE_prompt')
    for key in sorted(VE_prompt.keys()):
        print(f'{key}: {VE_prompt[key]}')
    print('BM_prompt')
    for key in sorted(BM_prompt.keys()):
        print(f'{key}: {BM_prompt[key]}')
    print(f'Total_num: {total_num} Omit_num: {len(omitted_img)}')
    print('Omitted images: ', omitted_img)
