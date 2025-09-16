import os
import torch
import torchvision
from templates import *        # Import model configuration templates
from renderer import *         # Import rendering utilities (if needed)
from experiment import LitModel # Import the Lightning model wrapper

# Select GPU device
device = 'cuda:0'

# Load configuration for HiNet model (128x128 resolution, ~130M parameters)
conf = ffhq128_ddpm_130M()

# Initialize the model and move it to GPU
model = LitModel(conf).to(device)

# Load pre-trained model checkpoint
state = torch.load(
    '',
    map_location='cpu'
)
model.load_state_dict(state['state_dict'])

# --- Sampling configuration ---
num_images = 10000   # total number of images to generate
batch_size = 16      # number of images per batch (adjust for your GPU memory)
output_dir = '~/CoprGuard/Generated/ffhq_ddim'
os.makedirs(output_dir, exist_ok=True)

# Counter for total saved images
saved_count = 0

# NOTE For Classifier-Free sampling
# label = torch.tensor([32], device=device)

while saved_count < num_images:
    # Generate Gaussian noise for this batch
    x_T = torch.randn((batch_size, 3, conf.img_size, conf.img_size), device=device)

    # Build the sampler
    sampler = conf._make_diffusion_conf(T=100).make_sampler()

    # Run DDIM sampling loop, only keep the last sample
    final_sample = None
    for sample in sampler.ddim_sample_loop_progressive(
        model=model.ema_model,
        noise=x_T
    ):
        final_sample = sample["sample"]
    
    # Run DDPM sampling loop, only keep the last sample
    # for sample in sampler.p_sample_loop_progressive(
    #                     model=model.ema_model,
    #                     noise=x_T):
        # final_sample = sample["sample"]
    
    # NOTE Run Classifier-Free sampling loop, only keep the last sample   
    # for sample in sampler.ddim_sample_loop_progressive(
    #     model=model.ema_model,
    #     noise=x_T,
    #     y=label
    # ):
        # final_sample = sample["sample"]
        
    # Rescale final output from [-1, 1] to [0, 1]
    images = (final_sample + 1) / 2  # Shape: [B, 3, H, W]

    # Save each image in the batch
    for img in images:
        if saved_count >= num_images:
            break  # stop once we reach target count
        torchvision.utils.save_image(img, f'{output_dir}/{saved_count:05d}.png')
        saved_count += 1

print(f"✅ Done! {saved_count} images saved to {output_dir}")
