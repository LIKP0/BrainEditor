import os
from glob import glob
import argparse
from PIL import Image
import numpy as np

# Use after test_cli.py (generate edited images)
# Convert to grey scale and rename + 0000
if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input_dir')
    # parser.add_argument('--output_dir')
    # args = parser.parse_args()
    # src_dir = args.input_dir
    # dst_dir = args.output_dir
    # pre_dir = '/data1/lijl/instruct-pix2pix/data/V11_stage1_CP_testbench/'
    # pre_dir = '/data1/lijl/instruct-pix2pix/data/V13_stage1_twoedit_testbench/'
    pre_dir = '/data1/lijl/instruct-pix2pix/data/OASIS2_dataset_1015FixBug_testbench'
    # src_dir = f'{pre_dir}/twoedit_data/'
    src_dir = f'{pre_dir}/twoedit_data/'
    dst_dir = f'{pre_dir}/nnUNet_input_twoedit/'
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    img_list = sorted(glob(src_dir + '**/*_edited.png', recursive=True))
    for img in img_list:
        index = img.split('/')[-1].split('.')[0]
        # index = img.split('/')[-1].split('.')[0][:-4]
        print(index)
        new_img = Image.open(img).convert('L')
        new_img.save(os.path.join(dst_dir, index + '_0000.png'))

    src_dir = '/data1/lijl/instruct-pix2pix/data/OASIS2_dataset_v8_pre2/nnUNet_pred_label/'
    test_dir = f'{pre_dir}/testdata/'
    dst_dir = f'{pre_dir}/test_structure_change_twoedit/'
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    for img_name in sorted(os.listdir(test_dir)):
        print(img_name)
        slice_index = img_name.split('XZ')[1].split('_')[0]
        origin_name = f'{img_name[:13]}_XZ{slice_index}.png'
        target_name = f'{img_name[:10] + img_name[13:16]}_XZ{slice_index}.png'
        origin_name = os.path.join(src_dir, origin_name)
        target_name = os.path.join(src_dir, target_name)

        if img_name[-2] == '-':  # negative class
            tmp = origin_name
            origin_name = target_name
            target_name = tmp

        before = f'{img_name}_before.png'
        after = f'{img_name}_target.png'
        cmd = f'cp {origin_name} {os.path.join(dst_dir, before)}'
        print(cmd)
        os.system(cmd)
        cmd = f'cp {target_name} {os.path.join(dst_dir, after)}'
        print(cmd)
        os.system(cmd)
