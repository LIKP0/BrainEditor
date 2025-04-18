import argparse
import os
import pickle
from glob import glob
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from PIL import Image
import numpy as np

if __name__ == '__main__':
    # each dir has before and after edit png
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input_dir')
    # args = parser.parse_args()
    # test_img_dir = args.input_dir

    test_img_dir = "/data2/jialin/instruct-pix2pix/data/OASIS2_dataset_1015FixBug_testbench/twoedit_data/"
    # test_img_dir = '/data1/lijl/instruct-pix2pix/data/V11_stage1_testbench/twoedit_data2/'
    # test_img_dir = '/data1/lijl/instruct-pix2pix/data/Ablation_study_v2/RealOne_testbench/twoedit_data/'
    # test_img_dir = '/data1/lijl/instruct-pix2pix/data/V11_stage1_CP_testbench/twoedit_data/'
    # test_img_dir = '/data1/lijl/TestResult_BIBM2024/Pix2Pix/twoedit_data/'
    # test_img_dir = '/data1/lijl/TestResult_BIBM2024/DiDiGAN/twoedit_data/'
    # test_img_dir = '/data1/lijl/TestResult_BIBM2024/StarGAN/twoedit_data/'
    # test_img_dir = '/data1/lijl/TestResult_BIBM2024/pSp2/twoedit_data/'
    test_list = sorted(os.listdir(test_img_dir))
    # heter_path = "/data1/lijl/instruct-pix2pix/data/heter_dict.pkl"  # specific for heterogeneous test set
    # with open(heter_path, 'rb') as f:
    #     heter_dict = pickle.load(f)

    test_class = ['V1C1', 'V2C1', 'V3C1']
    result1, result2, length = [], [], []
    total_avg_ps, total_avg_ss = 0, 0
    for cls in test_class:
        print(f'############# {cls} ##############')
        psnr = []
        ssim = []
        num = 0
        for i in range(len(test_list)):
            # MRI_ID = test_list[i][:13]
            # if not MRI_ID in heter_dict:
            #     continue
            postfix = test_list[i][-4:]
            if postfix == cls:
                print(test_list[i])
                origin_img = os.path.join(test_img_dir, test_list[i], f'{test_list[i]}_target.png')
                # origin_img = os.path.join(test_img_dir, test_list[i], f'{test_list[i]}_target.jpg')
                edit_img = os.path.join(test_img_dir, test_list[i], f'{test_list[i]}_before.png')
                # edit_img = os.path.join(test_img_dir, test_list[i], f'{test_list[i]}_edited.png')
                # edit_img = os.path.join(test_img_dir, test_list[i], f'{test_list[i]}_edit.png')
                origin = np.array(Image.open(origin_img).convert('L'))
                edit = np.array(Image.open(edit_img).convert('L'))
                origin = origin[:, 25:231]  # for 256
                edit = edit[:, 25:231]
                ps = compare_psnr(origin, edit)
                ss = compare_ssim(origin, edit, data_range=255)
                print(f'psnr: {ps:.2f}, ssim: {ss:.2f}')
                psnr.append(ps)
                ssim.append(ss)
                total_avg_ps += ps
                total_avg_ss += ss
                num += 1
        result1.append([round(np.mean(psnr), 3), round(np.std(psnr), 3)])
        result2.append([round(np.mean(ssim), 3), round(np.std(ssim), 3)])
        # result1.append([round(np.mean(psnr), 2), round(np.std(psnr), 2)])
        # result2.append([round(np.mean(ssim), 2), round(np.std(ssim), 2)])
        length.append(num)
        print(f'****** {cls} mean psnr(std): {np.mean(psnr):.2f}({np.std(psnr):.2f}) ******')
        print(f'****** {cls} mean ssim(std): {np.mean(ssim):.2f}({np.std(ssim):.2f}) ******')

    for i in range(len(test_class)):
        print(test_class[i])
        print('PSNR: ', result1[i])
        print('SSIM: ', result2[i])
        print('Length: ', length[i])
    print('Total Average PSNR: ', total_avg_ps / len(test_list))
    print('Total Average SSIM: ', total_avg_ss / len(test_list))
