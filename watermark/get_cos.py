import os
import glob
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import config as c
import modules.Unet_common as common
from model.hinet import Hinet
from utils.util import load_model, cleanup_state_dict, gauss_noise

# -----------------------------
# Utility Functions
# -----------------------------
def cosine_similarity_np(matrix1, matrix2):
    """
    Compute cosine similarity between two NumPy matrices.
    """
    v1 = matrix1.flatten().astype(np.float32)
    v2 = matrix2.flatten().astype(np.float32)
    dot_product = np.dot(v1, v2)
    return dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2))


# -----------------------------
# Dataset Definition
# -----------------------------
class MyDataset(Dataset):
    """
    Custom dataset for similarity evaluation.
    - Loads all PNG images from root_dir.
    - Uses watermark_path as a single fixed reference image.
    - Returns: (original image, horizontally flipped image, target image, image path)
    """
    def __init__(self, root_dir, watermark_path):
        self.items = glob.glob(f"{root_dir}/*.png")
        self.watermark_path = watermark_path

        self.transform_normal = T.Compose([
            T.Resize((128, 128)),
            T.ToTensor()
        ])

        self.transform_flip = T.Compose([
            T.RandomHorizontalFlip(p=1),
            T.Resize((128, 128)),
            T.ToTensor()
        ])

    def __getitem__(self, index):
        source_img = Image.open(self.items[index]).convert("RGB")
        target_img = Image.open(self.watermark_path).convert("RGB")
        return (
            self.transform_normal(source_img),  # original image
            self.transform_flip(source_img),   # horizontally flipped version
            self.transform_normal(target_img), # fixed target image
            self.items[index],                 # file path for reference
        )

    def __len__(self):
        return len(self.items)


def signle_cosine_similarity(matrix1, matrix2):
    vector1 = matrix1.flatten().astype(np.float32)
    vector2 = matrix2.flatten().astype(np.float32)
    dot_product = np.dot(vector1, vector2)

    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)

    similarity = dot_product / (norm_vector1 * norm_vector2)
    
    return similarity

# -----------------------------
# Main Process
# -----------------------------
if __name__ == "__main__":
    inc1, inc2, clamp = 3, 3, 2.0
    device = "cuda:4"
    watermark = "1.png"
    watermark_path = f"~/CoprGuard/watermark/wm_image/{watermark}"
    data_name = "FFHQ"
    root_dir = f"~/CoprGuard/Source/{data_name}"
    result_dir = "~/CoprGuard/watermark/results/similarities"
    ckpt_path = "~/CoprGuard/watermark/ckpts/naive.pt"

    # ======== Data Loading ========
    dataloader = DataLoader(
        MyDataset(os.path.expanduser(root_dir), os.path.expanduser(watermark_path)),
        batch_size=128,
        shuffle=False,
        num_workers=8,
    )

    # ======== Model Loading ========
    net = Hinet(clamp=clamp, inc1=inc1, inc2=inc2, n=8).to(device)
    load_model(net, os.path.expanduser(ckpt_path))
    net.eval()

    dwt = common.DWT().to(device)  # Discrete Wavelet Transform
    iwt = common.IWT().to(device)  # Inverse Wavelet Transform

    results = []
    paths = []

    # ======== Iteration over dataset ========
    for images, flipped_images, target_images, image_paths in dataloader:
        with torch.no_grad():
            # Send to GPU
            images = images.to(device)
            flipped_images = flipped_images.to(device)
            target_images = target_images.to(device)

            # Generate Gaussian noise for backward pass
            backward_z = gauss_noise((images.size(0), 12, 64, 64), device=device)

            # Define helper function to recover hidden secret from an image batch
            def recover_secret(input_images):
                """
                Perform:
                1. Apply DWT
                2. Concatenate with noise
                3. Run invertible network in reverse mode
                4. Apply IWT to reconstruct secret
                5. Flatten for cosine similarity computation
                """
                output_steg = dwt(input_images)
                output_rev = torch.cat((output_steg, backward_z), 1)
                backward_img = net(output_rev, rev=True)
                secret = backward_img.narrow(1, 4 * 3, backward_img.shape[1] - 4 * 3)
                return iwt(secret).view(secret.size(0), -1)

            # Recover secrets for original and flipped images
            secret_rev = recover_secret(images)
            secret_trev = recover_secret(flipped_images)
            target_images = target_images.view(target_images.size(0), -1)

            # Compute cosine similarity
            cc1 = torch.cosine_similarity(secret_rev, target_images)
            cc2 = torch.cosine_similarity(secret_trev, target_images)
            cc = torch.max(cc1, cc2)  # take the best match

            results.extend(cc.cpu().numpy().tolist())
            paths.extend(image_paths)

    # ======== Sort & Save Results ========
    sorted_pairs = sorted(zip(results, paths), key=lambda x: x[0])
    best_score, best_path = sorted_pairs[-1]

    print(f"Best matching image path: {best_path}")
    print(f"Highest similarity score: {best_score:.6f}")

    os.makedirs(os.path.expanduser(result_dir), exist_ok=True)
    with open(f"{os.path.expanduser(result_dir)}/{data_name}_{watermark}.txt", "w+") as f:
        f.write(str(sorted(results)))
