import torch.nn as nn
import torch
import sys
sys.path.append('..')
from modules.invblock import INV_block, INV_block_affine, INV_block_mid


class Hinet(nn.Module):
    """Hierarchical Invertible Network with 16 invertible blocks in two stages"""
    
    def __init__(self, clamp, inc1, inc2, n=8, use_affine=False):
        super().__init__()
        # Initialize blocks in two stages (8 blocks each)
        self.stage1 = nn.ModuleList([INV_block(clamp=clamp, inc1=inc1, inc2=inc2) for _ in range(n)])
        if use_affine:
            self.stage2 = nn.ModuleList([INV_block_affine(clamp=clamp, inc1=inc1, inc2=inc2) for _ in range(n)])
        else:
            self.stage2 = nn.ModuleList([INV_block(clamp=clamp, inc1=inc1, inc2=inc2) for _ in range(n)])
        
    def init_model(self):
        for key, param in self.named_parameters():
            split = key.split('.')
            if param.requires_grad:
                param.data = 0.01 * torch.randn(param.data.shape).to(param.device)
                if split[-2] == 'conv5':
                    param.data.fill_(0.)
                    
    def forward_stage1(self, x, rev=False):
        blocks = self.stage1
        if rev:
            blocks = reversed(blocks)
        for block in blocks:
            x = block(x, rev=rev)
        return x

    def forward_stage2(self, x, rev=False):
        blocks = self.stage2
        if rev:
            blocks = reversed(blocks)
        for block in blocks:
            x = block(x, rev=rev)
        return x

    def forward(self, x, rev=False):
        """Process input through invertible blocks in forward/reverse order"""
        # without affine_block
        blocks = self.stage1 + self.stage2
        if rev:
            blocks = reversed(blocks)
            
        for block in blocks:
            x = block(x, rev=rev)
            
        return x
    

class Hinet_mid(nn.Module):
    """Hierarchical Invertible Network with 16 invertible blocks in two stages"""
    
    def __init__(self, clamp, inc1, inc2, n=8, use_affine=False):
        super().__init__()
        # Initialize blocks in two stages (8 blocks each)
        self.stage1 = nn.ModuleList([INV_block_mid(clamp=clamp, inc1=inc1, inc2=inc2) for _ in range(n)])
        if use_affine:
            self.stage2 = nn.ModuleList([INV_block_affine(clamp=clamp, inc1=inc1, inc2=inc2) for _ in range(n)])
        else:
            self.stage2 = nn.ModuleList([INV_block_mid(clamp=clamp, inc1=inc1, inc2=inc2) for _ in range(n)])
        
    def init_model(self):
        for key, param in self.named_parameters():
            split = key.split('.')
            if param.requires_grad:
                param.data = 0.01 * torch.randn(param.data.shape).to(param.device)
                if split[-2] == 'conv5':
                    param.data.fill_(0.)
                    
    def forward_stage1(self, x, rev=False):
        blocks = self.stage1
        if rev:
            blocks = reversed(blocks)
        for block in blocks:
            x = block(x, rev=rev)
        return x

    def forward_stage2(self, x, rev=False):
        blocks = self.stage2
        if rev:
            blocks = reversed(blocks)
        for block in blocks:
            x = block(x, rev=rev)
        return x

    def forward(self, x, rev=False):
        """Process input through invertible blocks in forward/reverse order"""
        # without affine_block
        blocks = self.stage1 + self.stage2
        if rev:
            blocks = reversed(blocks)
            
        for block in blocks:
            x = block(x, rev=rev)
            
        return x


if __name__ == '__main__':
    import torch
    x = torch.rand((4, 48, 32, 32))
    net = Hinet(2.0, 4, 4, use_affine=True)
    # for block in net.stage1:
    #     x = block(x, rev=False)
    print(net.forward_stage2(x).size())