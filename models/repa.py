from torch import nn
import torch
import torch.nn.functional as F


class REPAHead(nn.Module):
    def __init__(self, in_channels_list=[320, 640], hidden_dim=512, out_dim=384):
        super().__init__()
        # First process each feature map through conv layers
        self.conv_projs = nn.ModuleList([
            nn.Sequential(
                # Simple conv block to process features
                nn.Conv2d(in_channels, hidden_dim, 1),
                nn.GELU(),
                nn.GroupNorm(32, hidden_dim),
                
                nn.Conv2d(hidden_dim, out_dim, 1),
            ) for in_channels in in_channels_list
        ])
        
        # MLP to process flattened features
        self.mlp = nn.Sequential(
            nn.Linear(out_dim * 256, hidden_dim),  # 256 is H*W after conv
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim * 256)  # Match DINOv2's flattened size
        )
    
    def forward(self, features):
        outputs = []
        for i, feature in enumerate(features):
            # Process each feature through its respective conv layer
            x = self.conv_projs[i](feature)
            
            # Resize all features to match the target size (16x16)
            target_size = (16, 16)  # Fixed size to match DINOv2 features
            if x.shape[-2:] != target_size:
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            
            outputs.append(x)

        # Now all tensors should have the same spatial dimensions
        avg_output = torch.stack(outputs).mean(dim=0)  # [B, 384, 16, 16]
        
        # Reshape to match DINOv2's output format
        B = avg_output.shape[0]
        avg_output = avg_output.reshape(B, -1)  # [B, 384*256]
        
        # Pass through MLP to match dimensions
        avg_output = self.mlp(avg_output)  # [B, 384*256]
        
        return avg_output
    