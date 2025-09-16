import torch
import torch.nn as nn
import torch.nn.functional as F

# Cross-Attention 模块（简化版）
class CrossAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_q = nn.Conv2d(dim, dim, 1)
        self.to_k = nn.Conv2d(dim, dim, 1)
        self.to_v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, A, B):
        B_q = self.to_q(A)
        B_k = self.to_k(B)
        B_v = self.to_v(B)

        B_, C, H, W = B_q.shape
        B_q = B_q.reshape(B_, self.heads, C // self.heads, H * W)
        B_k = B_k.reshape(B_, self.heads, C // self.heads, H * W)
        B_v = B_v.reshape(B_, self.heads, C // self.heads, H * W)

        attn = torch.einsum('bhcn,bhcm->bhnm', B_q, B_k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhnm,bhcm->bhcn', attn, B_v)
        out = out.reshape(B_, C, H, W)
        return self.proj(out) + A

# 可逆 Block（使用 additive coupling）
class InvertibleBlock(nn.Module):
    def __init__(self, channels, use_cross_attention=False):
        super().__init__()
        self.use_cross_attention = use_cross_attention
        self.split = channels // 2

        self.F = nn.Sequential(
            nn.Conv2d(self.split, self.split, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.split, self.split, 3, 1, 1)
        )
        self.G = nn.Sequential(
            nn.Conv2d(self.split, self.split, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(self.split, self.split, 3, 1, 1)
        )
        self.attn = CrossAttention(self.split) if use_cross_attention else nn.Identity()

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        y1 = x1 + self.F(x2)
        x2_enh = self.attn(x2, x1)  # 使用 cross-attention 增强 x2
        y2 = x2_enh + self.G(y1)
        return torch.cat([y1, y2], dim=1)

    def inverse(self, y):
        y1, y2 = torch.chunk(y, 2, dim=1)
        x2_enh = y2 - self.G(y1)
        x2 = self.attn(x2_enh, y1)  # 注意：在逆向也使用 cross-attention
        x1 = y1 - self.F(x2)
        return torch.cat([x1, x2], dim=1)

# 整体可逆网络结构
class InvertibleFeatureEnhancer(nn.Module):
    def __init__(self, channels=32, out_channels=48, num_blocks=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            InvertibleBlock(channels, use_cross_attention=(i % 2 == 0)) for i in range(num_blocks)
        ])
        self.final = nn.Conv2d(channels, out_channels, 1)  # 升维：B*32 → B*48

    def forward(self, A, B):
        x = torch.cat([A, B], dim=1)  # 输入 B×32×32×32
        for block in self.blocks:
            x = block(x)
        return self.final(x)  # 输出 B×48×32×32

    def inverse(self, enhanced):
        x = enhanced
        x = self.final.inverse(x) if hasattr(self.final, 'inverse') else x  # 若 final 可逆
        for block in reversed(self.blocks):
            x = block.inverse(x)
        A_rec, B_rec = torch.chunk(x, 2, dim=1)
        return A_rec, B_rec
