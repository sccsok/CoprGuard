import os
import glob
import argparse
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import config as c
from model.hinet import Hinet
import modules.Unet_common as common
from utils.util import load_model, gauss_noise
from modules.enhance import OLDUNet
from diffusers import AutoencoderKL


# -----------------------
# 1. Dataset
# -----------------------
class WatermarkDataset(Dataset):
    def __init__(self, root_dir, watermark_path, cropsize=128):
        self.items = sorted(glob.glob(f"{os.path.expanduser(root_dir)}/*.png"))
        self.watermark_path = watermark_path
        self.transform = T.Compose([
            T.Resize((cropsize, cropsize)),
            T.ToTensor()
        ])

    def __getitem__(self, index):
        cover_img = Image.open(self.items[index]).convert("RGB")
        secret_img = Image.open(self.watermark_path).convert("RGB")
        return (
            self.transform(cover_img),
            self.transform(secret_img),
            self.items[index],
        )

    def __len__(self):
        return len(self.items)


# -----------------------
# 2. Encode / Decode (Single)
# -----------------------
@torch.no_grad()
def encode_single(img_path, secret_path, out_path, net, dwt, iwt, transform, device):
    cover_image = Image.open(img_path).convert("RGB")
    cover_image = transform(cover_image).unsqueeze(0).to(device)
    secret_image = Image.open(secret_path).convert("RGB")
    secret_image = transform(secret_image).unsqueeze(0).to(device)

    cover_input = dwt(cover_image)
    secret_input = dwt(secret_image)
    input_img = torch.cat((cover_input, secret_input), dim=1)

    output = net(input_img)
    output_steg = output.narrow(1, 0, 4 * 3)
    steg_img = iwt(output_steg)
    
    torchvision.utils.save_image(steg_img, out_path)
    print(f"✅ Single image encoded and saved to {out_path}")


@torch.no_grad()
def decode_single(img_path, out_path, net, unet, vae, dwt, iwt, backward_z, transform, device):
    steg_image = Image.open(img_path).convert("RGB")
    steg_image = transform(steg_image).unsqueeze(0).to(device)
    
    if unet:
        steg_image = steg_image * 2 - 1
        latents = vae.encode(steg_image.to(dtype=torch.float32)).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        steg_image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
        steg_image = (steg_image + 1) / 2
        steg_image = unet(steg_image)

    output_steg = dwt(steg_image)
    output_rev = torch.cat((output_steg, backward_z), dim=1)
    backward_img = net(output_rev, rev=True)

    secret_rev = backward_img.narrow(1, 4 * 3, backward_img.shape[1] - 4 * 3)
    secret_rev = iwt(secret_rev)
    
    torchvision.utils.save_image(secret_rev, out_path)
    print(f"✅ Secret decoded and saved to {out_path}")


# -----------------------
# 3. Encode (Batch)
# -----------------------
@torch.no_grad()
def encode_batch(cover_batch, secret_batch, net, dwt, iwt):
    cover_batch = dwt(cover_batch)
    secret_batch = dwt(secret_batch)
    input_img = torch.cat((cover_batch, secret_batch), dim=1)
    output = net(input_img)
    steg_freq = output.narrow(1, 0, 4 * 3)
    steg_img = iwt(steg_freq)
    return steg_img


def batch_watermark(args, net, dwt, iwt, device):
    dataset = WatermarkDataset(args.root_dir, args.watermark_path, args.cropsize)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    os.makedirs(args.save_dir, exist_ok=True)

    for cover_batch, secret_batch, paths in dataloader:
        cover_batch, secret_batch = cover_batch.to(device), secret_batch.to(device)
        steg_batch = encode_batch(cover_batch, secret_batch, net, dwt, iwt)
        for i, p in enumerate(paths):
            save_path = os.path.join(args.save_dir, os.path.basename(p))
            torchvision.utils.save_image(steg_batch[i], save_path)

    print(f"✅ Finished embedding watermark for {len(dataset)} images. Saved to {args.save_dir}")


@torch.no_grad()
def decode_batch(steg_batch, net, dwt, iwt, backward_z, unet=None, vae=None):
    """Batch decode steg images to recover secrets."""
    if unet and vae:
        steg_batch = steg_batch * 2 - 1
        latents = vae.encode(steg_batch.to(dtype=torch.float32)).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        steg_batch = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
        steg_batch = (steg_batch + 1) / 2
        steg_batch = unet(steg_batch)

    output_steg = dwt(steg_batch)
    output_rev = torch.cat((output_steg, backward_z.expand(steg_batch.size(0), -1, -1, -1)), dim=1)
    backward_img = net(output_rev, rev=True)

    secret_rev = backward_img.narrow(1, 4 * 3, backward_img.shape[1] - 4 * 3)
    secret_rev = iwt(secret_rev)
    return secret_rev


def batch_dewatermark(args, net, dwt, iwt, backward_z, device, unet=None, vae=None):
    """Batch decode steg images from a folder and save recovered secrets."""
    steg_paths = sorted(glob.glob(f"{os.path.expanduser(args.steg_dir)}/*.png"))
    transform = T.Compose([
        T.Resize((args.cropsize, args.cropsize)),
        T.ToTensor(),
    ])
    dataloader = DataLoader(steg_paths, batch_size=args.batch_size, shuffle=False, num_workers=2)

    os.makedirs(args.save_dir, exist_ok=True)
    for batch_paths in dataloader:
        # load and stack images
        batch_imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            batch_imgs.append(transform(img))
        steg_batch = torch.stack(batch_imgs).to(device)

        secret_batch = decode_batch(steg_batch, net, dwt, iwt, backward_z, unet, vae)

        for i, p in enumerate(batch_paths):
            save_path = os.path.join(args.rev_dir, f"rev_{os.path.basename(p)}")
            torchvision.utils.save_image(secret_batch[i], save_path)

    print(f"✅ Finished decoding watermark for {len(steg_paths)} images. Saved to {args.rev_dir}")
    
# -----------------------
# 4. CLI Entry
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HiNet Watermark Embedding / Decoding")
    parser.add_argument("--root_dir", type=str, default="", help="Directory of cover images (for batch encode)")
    parser.add_argument("--watermark_path", type=str, default="", help="Path to secret image")
    parser.add_argument("--save_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--steg_dir", type=str, default="", help="Output directory")
    parser.add_argument("--rev_dir", type=str, default="", help="Output directory")
    parser.add_argument("--ckpt", type=str, default="~/CoprGuard/watermark/ckpts/model.pt")
    parser.add_argument("--cropsize", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--t2i", action="store_true", help="Enable OLDUNet pre/post-processing")

    # single image encode/decode
    args = parser.parse_args()

    # prepare device and models
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    if args.t2i:
        args.cropsize = 512
        args.ckpt = ""
    
    transform = T.Compose([
        T.Resize((args.cropsize, args.cropsize)),
        T.ToTensor(),
    ])
    
    net = Hinet(clamp=2.0, inc1=3, inc2=3, n=8).to(device).eval()
    unet = OLDUNet(3, 3, base_channels=64).to(device).eval() if args.t2i else None
    load_model(net, os.path.expanduser(args.ckpt), unet)

    vae = AutoencoderKL.from_pretrained('stabilityai/stable-diffusion-2-1', subfolder="vae").to(device)
    vae.requires_grad_(False)
    
    dwt, iwt = common.DWT().to(device), common.IWT().to(device)
    backward_z = gauss_noise((1, 12, args.cropsize // 2, args.cropsize // 2), device=device)
    
    # assert args.root_dir and args.watermark_path, "Batch mode needs --root_dir and --watermark_path"
    os.makedirs(args.save_dir, exist_ok=True)
    batch_watermark(args, net, dwt, iwt, device)
    # batch_dewatermark(args, net, dwt, iwt, backward_z, device, unet, vae)
    
    # watermarking and de-watermarking single image
    # img_dir = ""
    # secret_dir = ""
    # steg_dir = "
    # rev_dir = ""
    # encode_single(img_dir, secret_dir, steg_dir, net, dwt, iwt, transform, device)
    # decode_single(steg_dir, rev_dir, net, unet, vae, dwt, iwt, backward_z, transform, device)