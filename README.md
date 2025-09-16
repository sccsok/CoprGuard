# 🚀 Harnessing Frequency Spectrum Insights for Image Copyright Protection Against Diffusion Models
![GitHub Repo stars](https://img.shields.io/github/stars/sccsok/CoprGuard?style=social)
![GitHub forks](https://img.shields.io/github/forks/sccsok/CoprGuard?style=social)
![License](https://img.shields.io/github/license/sccsok/CoprGuard)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-🔥-red)

> 📌 **Paper:** [Harnessing Frequency Spectrum Insights for Image Copyright Protection Against Diffusion Model]([https://arxiv.org/abs/your-paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_Harnessing_Frequency_Spectrum_Insights_for_Image_Copyright_Protection_Against_Diffusion_CVPR_2025_paper.pdf))  
> 📖 **Conference:** CVPR 2025

<p align="center">
  <img src="images/teaser.png" alt="Teaser Image" width="650">
</p>

<!-- ## 🔥 Highlights
- 🎯 **State-of-the-art** performance on [Your Task].
- 💡 **Novel Approach** leveraging [key innovation].
- ⚡ **Efficient Implementation** using PyTorch & CUDA acceleration.
- 🔍 **Explainable Predictions** with [your method's unique property].--> 

# 📂 Project Structure
## Dataset Preparation
```bash
CoprGuard 
└── Source/ 
    ├── FFHQ/ 
    ├── CelebA-HQ/ 
    ├── BigGAN/ 
    ├── ... 
    └── Vggface/ 
└── HiNet/ 
    ├── FFHQ/ 
    ├── CelebA-HQ/ 
    ├── BigGAN/ 
    ├── ... 
    └── Vggface/ 
└── Generated/ 
    ├── FFHQ_DDIM/ 
    ├── BigGAN_DDIM/ 
    ├── ... 
    └── FFHQ_HiNet_DDIM/
```

# 🚀 Installation
```bash
# Clone the repository
git https://github.com/sccsok/CoprGuard.git
cd your-repo

# Create a virtual environment (optional) & install dependencies
conda env create -f environment.yml
```

# 🏋️‍♂️ Training & Evaluation

## 🖼️ Image Watermarking

- **Watermarking**

  **Download Pretrained Model**  
  Download [the pretrained HiNet](https://drive.google.com/drive/folders/1l3XBFYPMaNFdvCWyOHfB2qIPkpjIxZgE?usp=sharing) and put it in ```~/CoprGuard/watermark/ckpt```.

  **Apply Watermark to Training Images**  
  ```bash
  # For unconditional training images
  python wm.py --root_dir <> --watermark_path <> --save_dir <>
  ```

- **Generate Figure 6**

  **Compute Similarity Score**
  ```bash
  python get_cos.py
  ```

  **Plot Cosine Similarity Distribution**
  ```bash
  python plt.py
  ```

## 📌Unconditional Training & Evaluation

- **Prepare Dataset**  
  Please prepare the training dataset according to the **Dataset Preparation format**.

- **Train the DDPM Model**  
  Run the training scripts under `ddim/scripts`:
  ```bash
  cd ~/CoprGuard/ddim
  python scripts/xxx.py
  ```

- **Image Sampling**   
  Use the DDPM or DDIM scheduler for image sampling:
  ```bash
  python ddim/scripts/inference.py
  ```

- **Classifier-Free Training (Optional)**
  You may modify the DDIM code implementation or refer to the following repository: [classifier-free-diffusion-guidance-Pytorch](https://github.com/jcwang-gh/classifier-free-diffusion-guidance-Pytorch)

## 📊 Generate Figures
- **Generate Figure 1 & Figure 2 & Figure 10**
  ```bash
  cd ~/CoprGuard/frequence
  python frequency_analysis.py ~/CoprGuard/Source $WORKDIR/output <fft_hp/dct/...> --img-dirs <FFHQ BIgGAN ProGAN ImageNet> --log --vmin 1e-5 --vmax 1e-1
  python frequency_analysis.py ~/CoprGuard/Generated $WORKDIR/output <fft_hp/dct/...> --img-dirs <FFHQ BIgGAN ProGAN ImageNet> ImageNet_DDIM --log --vmin 1e-5 --vmax 1e-1
  ```

- **Generate Table 1**
  ```bash
  cd ~/CoprGuard/frequence
  python get_cos.py --type <fft/dct/...> --models1 <[FFHQ, xxx, ...]> --models2 <[FFHQ_DDIM, xxx, ...]> 
  ```

- **Generate Figure 3**
  ```bash
  cd ~/CoprGuard/frequence
  get_rapsd.ipynb
  ```

- **Generate Table 3**
  ```bash
  cd ~/CoprGuard/watermark
  python similarity_compute.py --mode folder --folder <> --watermark <> --resize 128 128
  ```

<!--# 📊 Results & Benchmark
## 🔬 Benchmark on [Your Dataset]
| Dataset | Method | Accuracy (%) | F1 Score |
|---------|--------|--------------|---------|
| YourDataset | **YourModel** | **95.2** | **0.89** |
| Baseline | XYZ Model | 90.1 | 0.85 |

<p align="center">
  <img src="assets/result.png" alt="Result Visualization" width="700">
</p>-->

## 📜 Citation
If you find our work useful, please consider citing:
```bibtex
@inproceedings{liu2025harnessing,
  title={Harnessing Frequency Spectrum Insights for Image Copyright Protection Against Diffusion Models},
  author={Liu, Zhenguang and Shuai, Chao and Fan, Shaojing and Dong, Ziping and Hu, Jinwu and Ba, Zhongjie and Ren, Kui},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={18653--18662},
  year={2025}
}
```

## 🙏 Acknowledgement
Our implementation benefits from the following open-source projects:
- [HiNet](https://github.com/TomTomTommi/HiNet)
- [Diffusion Autoencoders](https://github.com/konpatp/diffae.git)
- [Diffusion-model-deepfake-detection](https://github.com/jonasricker/diffusion-model-deepfake-detection.git)

We sincerely thank the authors for their great work.

---