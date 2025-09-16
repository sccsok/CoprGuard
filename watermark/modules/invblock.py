import torch
import torch.nn as nn
from modules.rrdb_denselayer import ResidualDenseBlock


class INV_block_mid(nn.Module):
    """Invertible block with coupling layer for reversible transformations"""
    
    def __init__(self, clamp=2.0, harr=True, inc1=3, inc2=3):
        super().__init__()
        # Initialize split dimensions based on Haar wavelet flag
        self.split_len1 = inc1 * 2 if harr else inc1
        self.split_len2 = inc2 * 2 if harr else inc2
        self.clamp = clamp
        
        # Transformation networks
        self.r = ResidualDenseBlock(self.split_len1, self.split_len2)  # ρ (scale)
        self.y = ResidualDenseBlock(self.split_len1, self.split_len2)  # η (translation)
        self.f = ResidualDenseBlock(self.split_len2, self.split_len1)  # φ (coupling)

    def e(self, s):
        """Exponential scaling function with sigmoid clamping"""
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, rev=False):
        """Forward/reverse pass through invertible block"""
        x1, x2 = x.split([self.split_len1, self.split_len2], dim=1)
        
        if not rev:
            # Forward transformation
            y1 = x1 + self.f(x2)
            y2 = self.e(self.r(y1)) * x2 + self.y(y1)
        else:
            # Reverse transformation
            y2 = (x2 - self.y(x1)) / self.e(self.r(x1))
            y1 = x1 - self.f(y2)
            
        return torch.cat((y1, y2), 1)
    
class INV_block(nn.Module):
    """Invertible block with coupling layer for reversible transformations"""
    
    def __init__(self, clamp=2.0, harr=True, inc1=3, inc2=3):
        super().__init__()
        # Initialize split dimensions based on Haar wavelet flag
        self.split_len1 = inc1 * 4 if harr else inc1
        self.split_len2 = inc2 * 4 if harr else inc2
        self.clamp = clamp
        
        # Transformation networks
        self.r = ResidualDenseBlock(self.split_len1, self.split_len2)  # ρ (scale)
        self.y = ResidualDenseBlock(self.split_len1, self.split_len2)  # η (translation)
        self.f = ResidualDenseBlock(self.split_len2, self.split_len1)  # φ (coupling)

    def e(self, s):
        """Exponential scaling function with sigmoid clamping"""
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, rev=False):
        """Forward/reverse pass through invertible block"""
        x1, x2 = x.split([self.split_len1, self.split_len2], dim=1)
        
        if not rev:
            # Forward transformation
            y1 = x1 + self.f(x2)
            y2 = self.e(self.r(y1)) * x2 + self.y(y1)
        else:
            # Reverse transformation
            y2 = (x2 - self.y(x1)) / self.e(self.r(x1))
            y1 = x1 - self.f(y2)
            
        return torch.cat((y1, y2), 1)

class INV_block_affine(nn.Module):
    """Enhanced invertible block with affine coupling and importance map"""
    
    def __init__(self, clamp=2.0, harr=True, inc1=3, inc2=3, imp_map=True):
        super().__init__()
        # Initialize dimensions and importance channels
        self.split_len1 = inc1 * 4 if harr else inc1
        self.split_len2 = inc2 * 4 if harr else inc2
        self.clamp = clamp
        self.imp = inc1 * 4 if imp_map else 0
        
        # Transformation networks with importance channels
        self.r = ResidualDenseBlock(self.split_len1 + self.imp, self.split_len2)  # ρ
        self.y = ResidualDenseBlock(self.split_len1 + self.imp, self.split_len2)  # η
        self.f = ResidualDenseBlock(self.split_len2, self.split_len1 + self.imp)  # φ
        self.p = ResidualDenseBlock(self.split_len2, self.split_len1 + self.imp)  # ψ

    def e(self, s):
        """Exponential scaling with sigmoid clamping"""
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, rev=False):
        """Forward/reverse pass with affine coupling"""
        x1, x2 = x.split([self.split_len1 + self.imp, self.split_len2], dim=1)
        if not rev:
            # Forward affine transformation
            y1 = self.e(self.p(x2)) * x1 + self.f(x2)
            y2 = self.e(self.r(y1)) * x2 + self.y(y1)
        else:
            # Reverse affine transformation
            y2 = (x2 - self.y(x1)) / self.e(self.r(x1))
            y1 = (x1 - self.f(y2)) / self.e(self.p(y2))
            
        return torch.cat((y1, y2), 1)

if __name__ == '__main__':
    # Test block with sample input
    net = INV_block(inc1=32, inc2=32)
    x = torch.rand((4, 256, 64, 64))
    print(net(x) - x)