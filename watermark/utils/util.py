import os
import logging
from datetime import datetime

import torch


def cleanup_state_dict(state_dict: dict) -> dict:
    """
    Clean the model's state_dict by removing 'module.' prefix
    (needed if the model was trained with DataParallel).
    """
    new_state = {}
    for k, v in state_dict.items():
        new_name = k[7:] if k.startswith("module.") else k
        new_state[new_name] = v
    return new_state


# def load_model(net: torch.nn.Module, ckpt_path: str):
#     """
#     Load model weights from checkpoint, removing temporary variables if present.
#     """
#     state_dicts = torch.load(ckpt_path)
#     network_state_dict = {k: v for k, v in state_dicts['model'].items() if 'tmp_var' not in k}
#     net.load_state_dict(cleanup_state_dict(network_state_dict))

def load_model(hinet: torch.nn.Module, ckpt_path: str, unet: torch.nn.Module=None):
    state_dicts = torch.load(ckpt_path, map_location='cpu')
    network_map = {'model': hinet, 'unet': unet}
    
    for key, network in network_map.items():
        if key in state_dicts and network is not None:
            state_dict = {
                k: v for k, v in state_dicts[key].items() 
                if 'tmp_var' not in k
            }
            state_dict = cleanup_state_dict(state_dict)
            network.load_state_dict(state_dict)
    
    
def gauss_noise(shape, device):
    """
    Generate Gaussian noise with the given shape and send to device.
    """
    noise = torch.zeros(shape, device=device)
    for i in range(noise.shape[0]):
        noise[i] = torch.randn_like(noise[i], device=device)
    return noise

 
def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)
        
        
from PIL import Image

def tiff_to_png(input_path, output_path):
    tiff_image = Image.open(input_path).resize((128, 128))
    tiff_image.save(output_path)


# input_path = "/home/shuaichao/HiNet/image/mandril_color.tif"
# output_path = "/home/shuaichao/HiNet/image/mandril_color_128.png"


# tiff_to_png(input_path, output_path)
