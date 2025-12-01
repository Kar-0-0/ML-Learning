import torch
import torch.nn as nn
import torch.nn.functional as F
import json


class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, norm_num_groups):
        super().__init__()
        self.norm1 = nn.GroupNorm(norm_num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(norm_num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.conv_shortcut = nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return self.conv_shortcut(x) + h


class DownEncoderBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, layers_per_block, norm_num_groups, add_downsample=True):
        super().__init__()
        self.resnets = nn.ModuleList()
        ch_in = in_channels
        for _ in range(layers_per_block):
            self.resnets.append(ResnetBlock2D(ch_in, out_channels, norm_num_groups))
            ch_in = out_channels
        self.add_downsample = add_downsample
        if add_downsample:
            self.downsamplers = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)])
        else:
            self.downsamplers = nn.ModuleList()

    def forward(self, x):
        for resnet in self.resnets:
            x = resnet(x)
        for downsampler in self.downsamplers:
            x = downsampler(x)
        return x


class UpDecoderBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, layers_per_block, norm_num_groups, add_upsample):
        super().__init__()
        self.resnets = nn.ModuleList()
        ch_in = in_channels
        for _ in range(layers_per_block):
            self.resnets.append(ResnetBlock2D(ch_in, out_channels, norm_num_groups))
            ch_in = out_channels
        if add_upsample:
            self.upsamplers = nn.ModuleList([nn.ConvTranspose2d(out_channels, out_channels, 4, stride=2, padding=1)])
        else:
            self.upsamplers = nn.ModuleList()

    def forward(self, x):
        for resnet in self.resnets:
            x = resnet(x)
        for upsampler in self.upsamplers:
            x = upsampler(x)
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
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

        # Downsampling
        self.down_blocks = nn.ModuleList()
        num_stages = len(block_out_channels)
        for i in range(num_stages):
            in_ch = block_out_channels[i-1] if i > 0 else block_out_channels[0]
            out_ch = block_out_channels[i]
            add_down = (i != num_stages-1)
            self.down_blocks.append(
                DownEncoderBlock2D(
                    in_ch, 
                    out_ch, 
                    layers_per_block, 
                    norm_num_groups,
                    add_down
                )
            )

        # Mid/Bottleneck block
        mid_ch = block_out_channels[-1]
        self.mid_block = ResnetBlock2D(mid_ch, mid_ch, norm_num_groups)

        # Latent Projection
        self.conv_mu = nn.Conv2d(mid_ch, latent_channels, kernel_size=3, padding=1)
        self.conv_log_var = nn.Conv2d(mid_ch, latent_channels, kernel_size=3, padding=1)
        
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
            out_channels,
            up_block_types, 
            block_out_channels,
            latent_channels,
            norm_num_groups,
            layers_per_block,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.conv_in = nn.Conv2d(latent_channels, block_out_channels[-1], kernel_size=3, padding=1)

        # Mid/Bottleneck block
        mid_ch = block_out_channels[-1]
        self.mid_block = ResnetBlock2D(mid_ch, mid_ch, norm_num_groups)

        # Up blocks
        self.up_blocks = nn.ModuleList()
        num_stages = len(block_out_channels)
        for i in reversed(range(num_stages)):
            in_channels = block_out_channels[i+1] if i < num_stages-1 else block_out_channels[-1]
            out_channels = block_out_channels[i]
            add_upsample = (i > 0)
            self.up_blocks.append(
                UpDecoderBlock2D(
                    in_channels,
                    out_channels,
                    layers_per_block,
                    norm_num_groups,
                    add_upsample
                )
            )
        self.conv_out = nn.Conv2d(block_out_channels[0], self.out_channels, 3, padding=1)
    
    def forward(self, x):
        x = self.conv_in(x)
        x = self.mid_block(x)

        for up_block in self.up_blocks:
            x = up_block(x)
        x = self.conv_out(x)

        return x


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
        self.scaling_factor = scaling_factor
        self.sample_size = sample_size
        self.encoder = EncoderKL(
            in_channels,
            latent_channels,
            block_out_channels,
            down_block_types,
            layers_per_block,
            norm_num_groups
        )
        self.decoder = DecoderKL(
            out_channels,
            up_block_types,
            block_out_channels,
            latent_channels,
            norm_num_groups,
            layers_per_block
        )
    
    def forward(self, x):
        mu, _ = self.encoder(x)
        z = mu
        x_recon = self.decoder(z)

        return x_recon




if __name__ == '__main__':
    from safetensors.torch import load_file
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]  # DL_STUDY
    vae_config_path = project_root / "configs" / "AutoKL.json"

    with open(vae_config_path, "r") as f:
        cfg = json.load(f)


    vae = AutoencoderKL(
    in_channels=cfg["in_channels"],
    out_channels=cfg["out_channels"],
    latent_channels=cfg["latent_channels"],
    block_out_channels=cfg["block_out_channels"],
    down_block_types=cfg["down_block_types"],
    up_block_types=cfg["up_block_types"],
    layers_per_block=cfg["layers_per_block"],
    norm_num_groups=cfg.get("norm_num_groups", 32),
    sample_size=cfg["sample_size"],
    scaling_factor=cfg["scaling_factor"],
    )
    vae.eval()

    project_root = Path(__file__).resolve().parent.parent
    vae_weights_path = project_root /  "model_weights" / "diffusion_pytorch_model.safetensors"
    sd = load_file(vae_weights_path, device="cpu")

    missing, unexpected = vae.load_state_dict(sd, strict=False)
    print("Missing:", missing)
    print("Unexpected:", unexpected)


