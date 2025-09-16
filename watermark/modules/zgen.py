import sys

import torch
import torch.nn as nn

sys.path.append('..')
import modules.module_util as mutil
from modules.rrdb_denselayer import ResidualDenseBlock

class CrossAttention(nn.Module):
    """Implements cross-attention mechanism between two feature maps.
    
    Args:
        input_channels (int): Number of channels in input feature map
        cond_channels (int): Number of channels in conditional feature map
        num_heads (int): Number of attention heads (currently unused, kept for compatibility)
    """
    def __init__(self, input_channels, cond_channels, num_heads=8):
        super().__init__()
        self.scale = (input_channels ** -0.5)
        
        # Feature transformation layers
        self.query = nn.Conv2d(input_channels, input_channels, kernel_size=1, bias=False)
        self.kv_transform = nn.Conv2d(cond_channels, input_channels*2, kernel_size=3, padding=1)
        self.projection = nn.Conv2d(input_channels, input_channels, kernel_size=1)

    def forward(self, x, condition):
        """Forward pass with residual connection.
        
        Args:
            x (Tensor): Input features [B, C, H, W]
            condition (Tensor): Conditional features [B, C_cond, H, W]
            
        Returns:
            Tensor: Attention-weighted features with residual connection
        """
        batch_size, channels, height, width = x.shape
        
        # Project features to query/key/value space
        q = self.query(x)
        k, v = self.kv_transform(condition).chunk(2, dim=1)
        
        # Compute attention weights
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = attn_weights.softmax(dim=-1)
        
        # Apply attention to value matrix
        attended_features = torch.matmul(attn_weights, v)
        return self.projection(attended_features) + x  # Residual connection


class MultiScaleCrossAttention(nn.Module):
    """Multi-scale cross-attention with feature downsampling.
    
    Args:
        base_channels (int): Base channel multiplier
        scale_factor (int): Feature scaling factor between stages
        cond_channels (int): Number of channels in conditional input
    """
    def __init__(self, base_channels, scale_factor, cond_channels):
        super().__init__()
        channels = base_channels * scale_factor
        self.scale = (channels ** -0.5)
        
        # Multi-scale feature processing
        self.query = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.kv_processor = nn.Sequential(
            nn.Conv2d(cond_channels, channels*2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(scale_factor),
            nn.Conv2d(channels*2, channels*2, kernel_size=3, padding=1)
        )
        self.projection = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x, condition):
        """Process features with multi-scale conditioning."""
        q = self.query(x)
        k, v = self.kv_processor(condition).chunk(2, dim=1)
        
        # Attention computation
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = attn_weights.softmax(dim=-1)
        
        return self.projection(attn_weights @ v) + x  # Residual connection


class UNetDecoderBlock(nn.Module):
    """UNet decoder block with upsampling and cross-attention.
    
    Args:
        in_channels (int): Input feature channels
        out_channels (int): Output feature channels
        cond_channels (int): Conditional feature channels
        is_final_layer (bool): Flag for final processing stage
    """
    def __init__(self,base_channels,
                ints,
                outs, cond_channels, is_final_layer=False):
        super().__init__()
        self.is_final_layer = is_final_layer
        
        # Upsampling and feature processing
        self.upsampler = nn.ConvTranspose2d(base_channels * ints, base_channels * outs, kernel_size=2, stride=2)
        self.feature_processor = nn.Sequential(
            nn.Conv2d(base_channels * outs,  base_channels * outs, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * outs,  base_channels * outs, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.attention = MultiScaleCrossAttention(base_channels, outs, cond_channels)

    def forward(self, x, condition):
        """Process features through decoder stage."""
        if not self.is_final_layer:
            x = self.upsampler(x)
            x = self.feature_processor[:2](x)  # First conv + ReLU
            x = self.attention(x, condition)  # Feature conditioning
            x = self.feature_processor[2:](x) # Second conv + ReLU
        else:
            x = self.feature_processor(x)
        return x


class ConditionalDecoder(nn.Module):
    """Hierarchical decoder with progressive upsampling and conditioning.
    
    Args:
        scale_stages (int): Number of upsampling stages
        noise_dim (int): Latent noise vector dimension
        cond_channels (int): Conditional feature channels
        base_channels (int): Base channel multiplier
    """
    def __init__(self, scale_stages=3, noise_dim=256, cond_channels=128, base_channels=64):
        super().__init__()
        # Initial feature projection
        self.init_projection = nn.Linear(noise_dim, base_channels*(2 ** scale_stages)*16)
        
        # Construct decoder stages
        self.decoder_stages = nn.ModuleList([
            UNetDecoderBlock(
                base_channels,
                ints = (2 ** (scale_stages - i)),
                outs = (2 ** (scale_stages - i - 1)),
                cond_channels=cond_channels,
                is_final_layer=(i == scale_stages)  # Only last block has last_layer=True
            )
            for i in range(scale_stages)
        ])
        
        self.final_conv = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)

    def forward(self, latent, condition):
        """Decode latent vector with conditional features."""
        # Initial feature generation
        x = self.init_projection(latent).view(latent.size(0), -1, 4, 4)
        
        # Progressive feature refinement
        for decoder_block in self.decoder_stages:
            x = decoder_block(x, condition)
            
        return self.final_conv(x)


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention module.
    
    Args:
        channels (int): Input feature channels
        reduction (int): Channel reduction ratio
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_pool = nn.AdaptiveAvgPool2d(1)
        self.attention_net = nn.Sequential(
            nn.Linear(channels, channels//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels//reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Enhance channel-wise features using attention weights."""
        batch_size, channels = x.size()[:2]
        weights = self.channel_pool(x).view(batch_size, channels)
        weights = self.attention_net(weights).view(batch_size, channels, 1, 1)
        return x * weights.expand_as(x)


class FeatureProcessor(nn.Module):
    """Dense feature processor with channel attention.
    
    Args:
        input_channels (int): Input feature channels
        output_channels (int): Output feature channels
    """
    def __init__(self, input_channels, output_channels=32):
        super().__init__()
        # Feature transformation layers
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(input_channels+output_channels, output_channels*2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(input_channels+3*output_channels, output_channels, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(inplace=True)
        self.channel_attention = ChannelAttention(input_channels + 3 * 32)
        
        # Initialize final layer weights
        mutil.initialize_weights([self.conv3], 0.)

    def forward(self, x):
        """Process features through dense connections."""
        x1 = self.activation(self.conv1(x))
        x2 = self.activation(self.conv2(torch.cat([x, x1], dim=1)))
        fused = torch.cat([x, x1, x2], dim=1)
        return self.conv3(self.channel_attention(fused))
    

class SecretFeatureEnhancer(nn.Module):
    def __init__(self, input_channels: int, output_channels: int = None) -> None:
        super().__init__()
        
        # Configure output channels with default value
        output_channels = 32 if output_channels is None else output_channels
        
        # Feature preprocessing blocks
        self.secret_preprocessor = FeatureProcessor(input_channels=input_channels,
            output_channels=output_channels
        )
        self.steg_preprocessor = FeatureProcessor(input_channels=input_channels,
            output_channels=output_channels
        )
        
        # Cross-modal attention module
        self.cross_attention = CrossAttention(input_channels=output_channels,
            cond_channels=output_channels
        )
        
        # Final feature refinement block
        self.feature_refiner = ResidualDenseBlock(input_channels=output_channels,
            output_channels=input_channels
        )

    def forward(self, secret_features: torch.Tensor, steg_features: torch.Tensor) -> torch.Tensor:
        # Preprocess both input streams
        preprocessed_secret = self.secret_preprocessor(secret_features)  # [B, 32, H, W]
        preprocessed_steg = self.steg_preprocessor(steg_features)        # [B, 32, H, W]
        
        # Cross-modal feature interaction
        attended_features = self.cross_attention(
            preprocessed_steg,
            preprocessed_secret
        )  # [B, 32, H, W]
        
        # Final feature refinement
        return self.feature_refiner(attended_features)
    
     
if __name__ == '__main__':
    import math
    x = torch.rand((1, 16, 16, 16))
    z = torch.rand(1, 128)

    net = ConditionalDecoder(scale_stages=int(math.log2(x.size(-1)//4)), noise_dim=128, cond_channels=16, base_channels=16)
    print(net(z, x).size())
    net2 = SecretFeatureEnhancer(input_channels=16)
    print(net2(x, x).size())