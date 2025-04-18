import argparse
from glob import glob
import math
from PIL import Image
import numpy as np


def compute_CX_change(seg1, seg2):
    Va = np.sum(seg1 == 1) + np.sum(seg1 == 2)
    Vb = np.sum(seg2 == 1) + np.sum(seg2 == 2)
    smooth = 1e-5
    # ratio = math.floor((Vb - Va) / (Va + smooth) * 100) * -1
    ratio = (Vb - Va) / (Va + smooth) * 100 * -1
    print(f'Cortex volume change: {ratio}%')
    return ratio


def compute_VX_change(seg1, seg2):
    Va = np.sum(seg1 == 3)
    Vb = np.sum(seg2 == 3)
    smooth = 1e-5
    ratio = (Vb - Va) / (Va + smooth) * 100
    print(f'Ventricle volume change: {ratio}%')
    return ratio


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input_dir')
    # args = parser.parse_args()
    # label_dir = args.input_dir
    label_dir = '/data1/lijl/instruct-pix2pix/data/Ablation_study/PossionOne_testbench/test_structure_change2/'
    label_list = sorted(glob(label_dir + '*_before.png'))

    skip_num = 0
    # interval = {'V0': [-5, 5], 'V1': [5, 15], 'V2': [15, 25], 'V3': [25, 35], 'V4': [35, 60], 'C0': [-4, 4], 'C1': [4, 7], 'C2': [7, 23]}
    # interval = {'V0': [-5, 5], 'V1': [5, 25], 'V2': [25, 60], 'C0': [-5, 7], 'C1': [7, 25]}
    # interval = {'V0': [-5, 5], 'V1': [5, 25], 'V2': [25, 50], 'C0': [-5, 3], 'C1': [3, 15]}
    # interval = {'V0': [-5, 5], 'V1': [5, 25], 'V2': [25, 100], 'C0': [-5, 0], 'C1': [0, 25]}
    # interval = {'V0': [0, 5], 'V1': [5, 25], 'V2': [25, 100], 'C0': [0, 25]}
    interval = {'V1': [0, 10], 'V2': [10, 20], 'V3': [20, 100], 'C1': [0, 25]}
    score = {'V0': [[], []], 'V1': [[], []], 'V2': [[], []], 'V3': [[], []], 'V4': [[], []], 'C0': [[], []],
             'C1': [[], []], 'C2': [[], []]}  # [ACC], [MAE]
    # interval = {'V1': [0, 10], 'V2': [10, 20], 'V3': [20, 100], 'C1': [0, 25], 'V-1': [-10, 0], 'V-2': [-20, -10],
    #             'V-3': [-100, -20], 'C-1': [-25, 0]}
    # score = {'V1': [[], []], 'V2': [[], []], 'V3': [[], []], 'V-1': [[], []], 'V-2': [[], []], 'V-3': [[], []],
    #          'C1': [[], []], 'C-1': [[], []]}

    test_ratio_change = []
    for origin_name in label_list:
        pred_name = origin_name[:-11] + '_edited.png'
        target_name = origin_name[:-11] + '_target.png'
        cls = origin_name.split('/')[-1].split('_')[4]
        print(origin_name, cls)
        origin = np.array(Image.open(origin_name))
        pred = np.array(Image.open(pred_name))
        target = np.array(Image.open(target_name))

        if cls[0] == 'V':
            print('Predict ratio: ', end='')
            ratio = compute_VX_change(origin, pred)
            print('GT ratio: ', end='')
            gt_ratio = compute_VX_change(origin, target)
        else:
            print('Predict ratio: ', end='')
            ratio = compute_CX_change(origin, pred)
            print('GT ratio: ', end='')
            gt_ratio = compute_CX_change(origin, target)

        lower, upper = interval[cls][0], interval[cls][1]
        if not lower <= gt_ratio < upper:
            print('WARNING: Ground truth class incorrect, skipped.')
            skip_num += 1
            continue

        if lower <= ratio < upper:
            score[cls][0].append(1)
            print('true')
        else:
            score[cls][0].append(0)
            print('false')
        score[cls][1].append(np.abs(gt_ratio - ratio))

    for cls in score.keys():
        print(cls, len(score[cls][0]))
        if len(score[cls][0]) > 0:
            print(f'{cls}: ACC {round(np.mean(score[cls][0]), 2)} MAE {round(np.mean(score[cls][1]), 2)}')
    print(f'{skip_num} samples skipped.')
