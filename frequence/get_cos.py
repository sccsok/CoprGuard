"""
Author: sccsok 95464654@qq.com
Date: 2023-08-21 15:07:04
LastEditors: sccsok chaoshuai@zju.edu.cn
LastEditTime: 2024-11-04 15:48:58
"""

import os
import numpy as np
import argparse


def aligned_cc(k1: np.ndarray, k2: np.ndarray) -> dict:
    """
    Aligned PRNU cross-correlation
    :param k1: (n1,nk) or (n1,nk1,nk2,...)
    :param k2: (n2,nk) or (n2,nk1,nk2,...)
    :return: {'cc':(n1,n2) cross-correlation matrix,'ncc':(n1,n2) normalized cross-correlation matrix}
    """

    # Type cast
    k1 = np.array(k1).astype(np.float32)
    k2 = np.array(k2).astype(np.float32)

    ndim1 = k1.ndim
    ndim2 = k2.ndim
    assert ndim1 == ndim2

    k1 = np.ascontiguousarray(k1).reshape(k1.shape[0], -1)
    k2 = np.ascontiguousarray(k2).reshape(k2.shape[0], -1)

    assert k1.shape[1] == k2.shape[1]

    k1_norm = np.linalg.norm(k1, ord=2, axis=1, keepdims=True)
    k2_norm = np.linalg.norm(k2, ord=2, axis=1, keepdims=True)

    k2t = np.ascontiguousarray(k2.transpose())

    cc = np.matmul(k1, k2t).astype(np.float32)
    ncc = (cc / (k1_norm * k2_norm.transpose())).astype(np.float32)

    return {'cc': cc, 'ncc': ncc}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute cross-correlation between frequency features")
    
    parser.add_argument(
        "--type",
        type=str,
        default="fft",
        help="Subfolder under ./output/frequency_analysis/features (default: fft)"
    )
    
    parser.add_argument(
        "--models1",
        nargs='+',
        default=['FFHQ'],
        help="List of model names for the first group"
    )
    
    parser.add_argument(
        "--models2",
        nargs='+',
        default=['FFHQ_DDIM'],
        help="List of model names for the second group"
    )

    args = parser.parse_args()

    root = "~/CoprGuard/metric/output/frequency_analysis/features"
    fft_dir = args.fft_dir
    MODELS = [args.models1, args.models2]

    # Load features for the first group
    f1 = [np.load(os.path.join(root, fft_dir, model_name + '.np.npy')) for model_name in MODELS[0]]
    # Load features for the second group
    f2 = [np.load(os.path.join(root, fft_dir, model_name + '.np.npy')) for model_name in MODELS[1]]

    f1, f2 = np.asarray(f1), np.asarray(f2)

    similarity = aligned_cc(f1, f2)['ncc']

    print(similarity)
