import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from templates import *

if __name__ == '__main__':
    gpus = [2,5]
    conf = stylegan2128_ddpm_130M()
    train(conf, gpus=gpus)

    conf.eval_programs = ['fid50']
    train(conf, gpus=gpus, mode='eval')