'''
name: 
msg: 
param: 
return: 
'''
from experiment import *


def ddpm():
    """
    base configuration for all DDIM-based models.
    """
    conf = TrainConfig()
    conf.batch_size = 8
    conf.beatgans_gen_type = GenerativeType.ddim
    # TODO revise
    # conf.beatgans_gen_type = GenerativeType.ddpm
    conf.beta_scheduler = 'linear'
    conf.data_name = 'ffhq'
    conf.diffusion_type = 'beatgans'
    conf.eval_ema_every_samples = 200_000
    conf.eval_every_samples = 200_000
    conf.fp16 = True
    conf.lr = 1e-4
    conf.model_name = ModelName.beatgans_ddpm
    conf.net_attn = (16, )
    conf.net_beatgans_attn_head = 1
    conf.net_beatgans_embed_channels = 512
    conf.net_ch_mult = (1, 2, 4, 8)
    conf.net_ch = 64
    conf.sample_size = 32
    conf.T_eval = 100  #
    conf.T = 1000
    conf.make_model_conf()
    return conf


def ffhq64_ddpm():
    conf = ddpm()
    conf.data_name = 'ffhqlmdb256'
    conf.warmup = 0
    conf.total_samples = 72_000_000
    conf.scale_up_gpus(4)
    return conf


def celeba64d2c_ddpm():
    conf = ffhq128_ddpm()
    conf.data_name = 'celebalmdb'
    conf.eval_every_samples = 10_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.total_samples = 72_000_000
    conf.name = 'celeba64d2c_ddpm'
    return conf


def ffhq128_ddpm():
    conf = ddpm()
    conf.data_name = 'ffhqlmdb256'
    conf.warmup = 0
    conf.total_samples = 48_000_000
    conf.img_size = 128
    conf.net_ch = 128
    # channels:
    # 3 => 128 * 1 => 128 * 1 => 128 * 2 => 128 * 3 => 128 * 4
    # sizes:
    # 128 => 128 => 64 => 32 => 16 => 8
    conf.net_ch_mult = (1, 1, 2, 3, 4)
    conf.eval_every_samples = 1_000_000
    conf.eval_ema_every_samples = 1_000_000
    conf.scale_up_gpus(4)
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.make_model_conf()
    return conf

def ffhq256_ddpm():
    conf = ddpm()
    conf.data_name = 'ffhqlmdb256'
    conf.warmup = 0
    conf.total_samples = 48_000_000
    conf.img_size = 256
    conf.net_ch = 128
    # channels:
    # 3 => 128 * 1 => 128 * 1 => 128 * 2 => 128 * 3 => 128 * 4
    # sizes:
    # 128 => 128 => 64 => 32 => 16 => 8
    conf.net_ch_mult = (1, 1, 2, 3, 4)
    conf.eval_every_samples = 1_000_000
    conf.eval_ema_every_samples = 1_000_000
    conf.scale_up_gpus(4)
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.make_model_conf()
    return conf


def ffhq128_ddpm_72M():
    conf = ffhq128_ddpm()
    conf.total_samples = 72_000_000
    conf.name = 'ffhq128_ddpm_72M'
    return conf


def vgg128_ddpm_130M():
    conf = ffhq128_ddpm()
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.name = 'vgg128_ddpm_130M'
    return conf


def ffhq_diagnosis_merge_0_5_ddpm_130M():
    conf = ffhq128_ddpm()
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.name = 'ffhq_diagnosis_merge_0_5_ddpm_130M'
    return conf


def ffhq_diagnosis_merge_0_8_ddpm_130M():
    conf = ffhq128_ddpm()
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.name = 'ffhq_diagnosis_merge_0_8_ddpm_130M'
    return conf


def ffhq_diagnosis_merge_5_100_ddpm_130M():
    conf = ffhq128_ddpm()
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.name = 'ffhq_diagnosis_merge_5_100_ddpm_130M'
    return conf
    
    
def ffhq_stylegan2_merge_0_5_ddpm_130M():
    conf = ffhq128_ddpm()
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.name = 'ffhq_stylegan2_merge_0_5_ddpm_130M'
    return conf

def ffhq_hinet_merge_1_100_ddpm_130M():
    conf = ffhq128_ddpm()
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.name = 'ffhq_hinet_merge_1_100_ddpm_130M'
    return conf

def ffhq_stylegan2_merge_0_8_ddpm_130M():
    conf = ffhq128_ddpm()
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.name = 'ffhq_stylegan2_merge_0_8_ddpm_130M'
    return conf

def celeb128_ft_ddpm_130M():
    conf = ffhq128_ddpm()
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.name = 'celeb128_ft_ddpm_130M'
    return conf

def hinet128_ft_ddpm_130M():
    conf = ffhq128_ddpm()
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.name = 'hinet128_ft_ddpm_130M'
    return conf
    
def progan128_ddpm_130M():
    conf = ffhq128_ddpm()
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.name = 'progan128_ddpm_130M'
    return conf

def biggan128_ddpm_130M():
    conf = ffhq128_ddpm()
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.name = 'biggan128_ddpm_130M'
    return conf


def ffhq128_ddpm_130M():
    conf = ffhq128_ddpm()
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.name = 'ffhq128_ddpm_130M'
    return conf


def diagnosis128_ddpm_130M():
    conf = ffhq128_ddpm()
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.name = 'diagnosis128_ddpm_130M'
    return conf


def hinet128_ddpm_130M():
    conf = ffhq128_ddpm()
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.name = 'hinet128_ddpm_130M'
    return conf

def stylegan2128_ddpm_130M():
    conf = ffhq128_ddpm()
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.name = 'stylegan2128_ddpm_130M'
    return conf


def horse128_ddpm():
    conf = ffhq128_ddpm()
    conf.data_name = 'horse256'
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.name = 'horse128_ddpm'
    return conf


def bedroom128_ddpm():
    conf = ffhq128_ddpm()
    conf.data_name = 'bedroom256'
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.total_samples = 120_000_000
    conf.name = 'lsun128_ddpm_130M'
    return conf


def bedroom256_ddpm():
    conf = ffhq256_ddpm()
    conf.data_name = 'bedroom256'
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.total_samples = 120_000_000
    conf.name = 'bedroom256_ddpm'
    return conf
