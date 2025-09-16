'''
calculate the PSNR and SSIM.
same as MATLAB's results
'''
import os
import math
import numpy as np
import cv2
import glob
from natsort import natsorted
from skimage.metrics import structural_similarity as ssim

def main():
    # Configurations

    # GT - Ground-truth;
    # Gen: Generated / Restored / Recovered images
    folder_GT = ''
    folder_Gen = ''

    PSNR_all = []
    SSIM_all = []
    img_list = sorted(glob.glob(folder_GT + '/*'))
    img_list = natsorted(img_list)

    for i, img_path in enumerate(img_list):
        if not (img_path.endswith('.png') or img_path.endswith('.JPEG')):
            continue
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        # base_name = base_name[:5]
        im_GT = cv2.resize(cv2.imread(img_path), (512, 512))
        # print(base_name)
        # print(img_path)
        # print(os.path.join(folder_Gen, base_name + '.png'))
        im_Gen = cv2.resize(cv2.imread(os.path.join(folder_Gen, base_name + '.png')), (512, 512))

        PSNR = calculate_psnr(im_GT, im_Gen)

        SSIM = calculate_ssim(im_GT, im_Gen)
        print('{:3d} - {:25}. \tPSNR: {:.6f} dB, \tSSIM: {:.6f}'.format(
            i + 1, base_name, PSNR, SSIM))
        PSNR_all.append(PSNR)
        SSIM_all.append(SSIM)
        if  (1 + 1) % 10000 == 0:
            print('{:3d} Average: PSNR: {:.6f} dB, SSIM: {:.6f}'.format(
                1 + 1, sum(PSNR_all) / len(PSNR_all),
                sum(SSIM_all) / len(SSIM_all)))
    print('Average: PSNR: {:.6f} dB, SSIM: {:.6f}'.format(
        sum(PSNR_all) / len(PSNR_all),
        sum(SSIM_all) / len(SSIM_all)))

    with open('1.txt', 'w') as f:
        f.write(str(PSNR_all))


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(image1, image2):
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    (score, _) = ssim(gray_image1, gray_image2, full=True)
    return score


def bgr2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


if __name__ == '__main__':
    main()
