import torch
import torch.nn as nn
import modules.module_util as mutil

class ResidualDenseBlock(nn.Module):
    """Residual Dense Block combining DenseNet and ResNet concepts[2,4](@ref)
    
    Args:
        input (int): Number of input channels
        output (int): Number of output channels
        bias (bool): Whether to use bias in convolution layers
    """
    
    def __init__(self, input_channels, output_channels, bias=True):
        super().__init__()
        self.growth_channels = 32  # Intermediate channels in dense connections[2](@ref)
        
        # 5 convolutional layers with increasing input channels[2](@ref)
        self.conv1 = nn.Conv2d(input_channels, self.growth_channels, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(input_channels + self.growth_channels, self.growth_channels, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(input_channels + 2*self.growth_channels, self.growth_channels, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(input_channels + 3*self.growth_channels, self.growth_channels, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(input_channels + 4*self.growth_channels, output_channels, 3, 1, 1, bias=bias)
        
        self.lrelu = nn.LeakyReLU(inplace=True)
        mutil.initialize_weights([self.conv5], 0.)  # Initialize final conv weights to 0[2](@ref)

    def forward(self, x):
        """Forward pass with dense connections and local residual[2,4](@ref)
        
        Args:
            x (Tensor): Input tensor of shape (N, C, H, W)
            
        Returns:
            Tensor: Output tensor with local residual connection
        """
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        
        # Local residual connection (implicit through concatenation)[4](@ref)
        return x5  