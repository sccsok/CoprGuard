import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from natsort import natsorted

def to_rgb(image):
    """Convert image to RGB format if not already"""
    return image.convert('RGB') if image.mode != 'RGB' else image


class Hinet_VAE_Dataset(Dataset):
    """Paired dataset class for HiNet VAE training"""
    def __init__(self, transform=None, mode="train", cfg=None):
        self.transform = transform
        self.mode = mode
        path_pattern = cfg.TRAIN_PATH if mode == 'train' else cfg.VAL_PATH
        ext = cfg.format_train if mode == 'train' else cfg.format_val
        self.files = natsorted(glob.glob(f"{path_pattern}/*.{ext}"))[:5000 if mode == 'train' else None]

    def __getitem__(self, index):
        try:
            cover = self.transform(to_rgb(Image.open(self.files[index])))
            secret = self.transform(to_rgb(Image.open(self.files[-index-1])))
            return cover, secret
        except:
            return self.__getitem__(index + 1)

    def __len__(self):
        return len(self.files)
