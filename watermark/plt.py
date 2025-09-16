import matplotlib.pyplot as plt
import numpy as np
import json
import os


def load_and_pad_data(file_path, target_size=60000):
    """
    Load data from a JSON file and pad it to the target size if it has fewer samples.
    
    Parameters
    ----------
    file_path : str
        Path to the JSON file containing numeric data.
    target_size : int, optional
        Number of samples to ensure. If data is smaller, random samples are duplicated.

    Returns
    -------
    np.ndarray
        A float64 numpy array with shape (target_size,).
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    data = np.asarray(data).astype(np.float64)

    # Randomly pad if data is too small
    if data.shape[0] < target_size:
        pad_idx = np.random.randint(0, len(data), target_size - len(data))
        data = np.concatenate((data, data[pad_idx]))

    return data


def plot_histogram(file_path, rgb=(243, 114, 82), label=None, bins=100):
    """
    Load data, pad if needed, and plot a histogram on the current matplotlib axes.
    
    Parameters
    ----------
    file_path : str
        Path to the JSON file.
    rgb : tuple
        RGB color tuple (0-255 range) for histogram color.
    label : str, optional
        Legend label for this dataset.
    bins : int, optional
        Number of histogram bins.
    """
    data = load_and_pad_data(file_path)
    hex_color = '#%02x%02x%02x' % rgb  # Convert RGB to hex format
    plt.hist(data, bins=bins, alpha=0.5, color=hex_color, label=label)


# rgbs = [(254, 210, 151), (252, 180, 97), (247, 147, 80), (171, 224, 228), (127, 198, 211), (111, 120, 185)]
if __name__ == "__main__":
    # ----------------------------
    # Matplotlib global settings
    # ----------------------------
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 10

    # Dataset names and result files
    data_names = ["FFHQ", "LSUN", "ImageNet"]
    watermark = "1.png"  # Empty string means no extra suffix in filename

    # File paths (expand ~ to absolute path)
    file_paths = [
        os.path.expanduser(f"~/CoprGuard/watermark/results/similarities/{name}_{watermark}.txt")
        for name in data_names
    ]

    # RGB colors for each dataset
    colors = [(151, 179, 25), (229, 139, 123), (229, 139, 123)]
    labels = ["R=100%", "R=50%", "R=20%"]

# ----------------------------
    # Plot histograms
    # ----------------------------
    plt.figure(figsize=(6, 4))
    for file_path, color, label in zip(file_paths, colors, labels):
        plot_histogram(file_path, rgb=color, label=label)

    # Configure plot appearance
    plt.xlim(0.65, 1.0)
    plt.legend(loc='upper left')
    plt.title('Cosine Similarity Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(False)

    # Save plot to file
    plt.savefig('./results/plt.png')
    plt.close()
