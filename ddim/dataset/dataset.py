import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd

import torchvision.transforms.functional as Ftrans


class ALL_Dataset(Dataset):
    def __init__(self, 
                 method,
                 mix=0,
                 image_size=128,
                 data_dir=None,
                 original_resolution=256,
                 split=None,
                 as_tensor: bool = True,
                 do_augment: bool = False,
                 do_normalize: bool = True,
                 **kwargs
                 ):
        
        transform = [
            transforms.Resize((image_size, image_size)),
        ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if as_tensor:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)
        # self.method = method
        
        exts = ['png', 'jpg']
        count = 600000
        if isinstance(method, list):
            path1 = [p for ext in exts for p in Path(f'{os.path.join(data_dir, method[0])}').glob(f'*.{ext}')]
            path2 = [p for ext in exts for p in Path(f'{os.path.join(data_dir, method[1])}').glob(f'*.{ext}')]
            self.paths = path1[0:int(count * mix)] + path2[0:count - int(count * mix)]
        else:
            self.paths = [p for ext in exts for p in Path(f'{os.path.join(data_dir, method)}').glob(f'*.{ext}')]
            self.paths = self.paths[:count]
            
        self.length = len(self.paths)
    
        print(f"Training dataset is {method}, mix is {mix, 1 - mix}: {self.length}")
    
    def __getitem__(self, index):
        assert index < self.length
        img = Image.open(self.paths[index]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            
        return {'img': img, 'index': index}

    def __len__(self):
        
        return self.length