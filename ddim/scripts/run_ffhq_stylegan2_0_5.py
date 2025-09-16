import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from templates import *

if __name__ == '__main__':
    gpus = [4,5]
    conf = ffhq_stylegan2_merge_0_5_ddpm_130M()
    # batch-size = 16 for single 3090-gpu (14G)
    # batch-size = 24
    conf.batch_size = 48
    # train(conf, gpus=gpus)

    gpus = [0,1]
    conf.eval_programs = ['fid100']
    train(conf, gpus=gpus, mode='eval')