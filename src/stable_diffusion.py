import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path


class ResnetBlock2D(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels, 
            norm_num_groups, 
    ):
        super().__init__()
        
        self.group_norm1 = nn.GroupNorm(norm_num_groups, in_channels)
        self.silu1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.group_norm2 = nn.GroupNorm(norm_num_groups, out_channels)
        self.silu2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
        
    def forward(self, x):
        h = self.conv1(self.silu1(self.group_norm1(x)))
        h = self.conv2(self.silu2(self.group_norm2(h)))
        
        return self.skip(x)


class DownEncoderBlock2d(nn.Module):
    def __init__(
      self, 
      in_channels, 
      out_channels,
      layers_per_block,
      norm_num_groups, 
      add_downsample=True
    ):
        super().__init__()
        self.resnets = nn.ModuleList()
        for _ in range(layers_per_block):
            self.resnets.append(ResnetBlock2D(in_channels, out_channels, norm_num_groups))
        self.add_downsample = add_downsample
        if add_downsample:
            self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=2)
    
    def forward(self, x):
        for resnet in self.resnets:
            x = resnet(x)
        if self.add_downsample:
            x = self.downsample(x)
        
        return x

class EncoderKL(nn.Module):
    def __init__(
        self, 
        in_channels,
        latent_channels, 
        block_out_channels, 
        down_block_types, 
        layers_per_block,
        norm_num_groups
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(3, block_out_channels[0], kernel_size=3, padding=1)

        # Downsampling
        self.down_blocks = nn.ModuleList()
        num_stages = len(block_out_channels)
        for i in range(num_stages):
            in_ch = block_out_channels[i-1] if i > 0 else block_out_channels[0]
            out_ch = block_out_channels[i]
            add_down = (i != num_stages-1)
            self.down_blocks.append(
                DownEncoderBlock2d(
                    in_ch, 
                    out_ch, 
                    layers_per_block, 
                    norm_num_groups,
                    add_down
                )
            )

            # Mid/Bottleneck block
            mid_ch = block_out_channels[-1]
            self.mid_block = nn.Conv2d(mid_ch, mid_ch, norm_num_groups)

            # Latent Projection
            self.conv_mu = nn.Conv2d(mid_ch, latent_channels)
            self.conv_log_var = nn.Conv2d(mid_ch, latent_channels)
        
    def forward(self, x):
        x = self.conv_in(x)

        for down_block in self.down_blocks:
            x = down_block(x)

        x = self.mid_block(x)
        mu = self.conv_mu(x)
        log_var = self.conv_log_var(x)

        return mu, log_var

    

class DecoderKL(nn.Module):
    def __init__(
            self, 
            up_block_types, 
            block_out_channels
    ):
        super().__init__()


class AutoencoderKL(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        latent_channels,
        block_out_channels,
        down_block_types,
        up_block_types,
        layers_per_block,
        norm_num_groups,
        sample_size,
        scaling_factor,
    ):
        super().__init__()




if __name__ == '__main__':
    project_root = Path(__file__).resolve().parents[1]  # DL_STUDY
    vae_config_path = project_root / "configs" / "AutoKL.json"

    with open(vae_config_path, "r") as f:
        vae_cfg = json.load(f)


