import cv2
import numpy as np
import os

if __name__ == '__main__':

    pre = '/data1/lijl/instruct-pix2pix/data/V11_stage1_testbench/testdata'
    index_list = ['OAS2_0002_MR2MR3_XZ76_C1', 'OAS2_0002_MR2MR3_XZ76_V2', 'OAS2_0002_MR2MR3_XZ86_C1',
                  'OAS2_0002_MR2MR3_XZ86_V2', 'OAS2_0013_MR2MR3_XZ72_C1', 'OAS2_0013_MR2MR3_XZ72_V1']
    for index in index_list:
        # save_dir = '/data1/lijl/TestResult_BIBM2024/Figure2/'
        source1 = f'{pre}/{index}/0_0.jpg'
        source2 = f'{pre}/{index}/0_1.jpg'
        dst_path = f'{pre}/{index}/{index}_heatmap.png'
        print(source1, source2, dst_path)
        image1 = cv2.imread(source1)
        image2 = cv2.imread(source2)

        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        kernel_gaussian = cv2.getGaussianKernel(3, 0) * cv2.getGaussianKernel(3, 0).T
        diff_simple = cv2.filter2D(gray1.astype(np.float32) - gray2.astype(np.float32), -1, kernel_gaussian)
        abs_diff = np.abs(diff_simple)

        norm_diff = cv2.normalize(abs_diff, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(norm_diff.astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(dst_path, heatmap)
        print(f'Saved to {dst_path}')
