import torch
import torch.nn as nn

from model import get_time_embedding


def _num_groups(channels: int) -> int:
    return min(8, channels)


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv_first = nn.Sequential(
            nn.GroupNorm(_num_groups(in_channels), in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.time_emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )
        self.conv_second = nn.Sequential(
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, time_emb):
        residual = self.residual_conv(x)
        out = self.conv_first(x)
        out = out + self.time_emb_layers(time_emb)[:, :, None, None]
        out = self.conv_second(out)
        return out + residual


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.norm = nn.GroupNorm(_num_groups(channels), channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        tokens = x.reshape(batch_size, channels, height * width)
        tokens = self.norm(tokens).transpose(1, 2)
        attn_out = self.attention(tokens, tokens, tokens, need_weights=False)[0]
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, channels, height, width)
        return x + attn_out


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        time_emb_dim,
        down_sample,
        num_heads,
        num_res_blocks=2,
    ):
        super().__init__()
        blocks = []
        channels = in_channels
        for _ in range(num_res_blocks):
            blocks.append(ResNetBlock(channels, out_channels, time_emb_dim))
            channels = out_channels
        self.res_blocks = nn.ModuleList(blocks)
        self.attention = AttentionBlock(out_channels, num_heads)
        self.down_sample_conv = (
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
            if down_sample
            else nn.Identity()
        )

    def forward(self, x, time_emb):
        out = x
        for block in self.res_blocks:
            out = block(out, time_emb)
        out = self.attention(out)
        skip = out
        out = self.down_sample_conv(out)
        return out, skip


class MidBlock(nn.Module):
    def __init__(self, channels, time_emb_dim, num_heads, num_res_blocks=2):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResNetBlock(channels, channels, time_emb_dim)
            for _ in range(num_res_blocks)
        ])
        self.attention = AttentionBlock(channels, num_heads)

    def forward(self, x, time_emb):
        out = x
        for block in self.res_blocks:
            out = block(out, time_emb)
        out = self.attention(out)
        return out


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        time_emb_dim,
        up_sample,
        num_heads,
        num_res_blocks=2,
    ):
        super().__init__()
        self.up_sample_conv = (
            nn.ConvTranspose2d(
                in_channels // 2,
                in_channels // 2,
                kernel_size=4,
                stride=2,
                padding=1,
            )
            if up_sample
            else nn.Identity()
        )
        blocks = []
        for i in range(num_res_blocks):
            block_in = in_channels if i == 0 else out_channels
            blocks.append(ResNetBlock(block_in, out_channels, time_emb_dim))
        self.res_blocks = nn.ModuleList(blocks)
        self.attention = AttentionBlock(out_channels, num_heads)

    def forward(self, x, skip, time_emb):
        x = self.up_sample_conv(x)
        x = torch.cat([x, skip], dim=1)
        out = x
        for block in self.res_blocks:
            out = block(out, time_emb)
        out = self.attention(out)
        return out


class UnetLarge(nn.Module):
    def __init__(self, in_channels, num_res_blocks=2, num_mid_blocks=2):
        super().__init__()
        self.down_channels = [64, 128, 256, 512]
        self.time_emb_dim = 256
        self.down_sample = [True, True, True]
        self.num_res_blocks = num_res_blocks
        self.num_mid_blocks = num_mid_blocks
        self.out_channels = 64
        num_heads = 8

        self.time_proj = nn.Sequential(
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
            nn.SiLU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
        )
        self.conv_in = nn.Conv2d(in_channels, self.down_channels[0], kernel_size=3, padding=1)

        self.downs = nn.ModuleList([
            DownBlock(
                self.down_channels[i],
                self.down_channels[i + 1],
                self.time_emb_dim,
                self.down_sample[i],
                num_heads,
                num_res_blocks,
            )
            for i in range(len(self.down_channels) - 1)
        ])

        bottleneck_channels = self.down_channels[-1]
        self.mids = nn.ModuleList([
            MidBlock(bottleneck_channels, self.time_emb_dim, num_heads, num_res_blocks)
            for _ in range(num_mid_blocks)
        ])

        self.ups = nn.ModuleList()
        for i in reversed(range(len(self.down_channels) - 1)):
            out_ch = self.down_channels[i] if i > 0 else self.out_channels
            in_ch = self.down_channels[i + 1] * 2
            self.ups.append(
                UpBlock(
                    in_ch,
                    out_ch,
                    self.time_emb_dim,
                    self.down_sample[i],
                    num_heads,
                    num_res_blocks,
                )
            )

        self.norm_out = nn.GroupNorm(_num_groups(self.out_channels), self.out_channels)
        self.conv_out = nn.Conv2d(self.out_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x, time):
        out = self.conv_in(x)
        time_emb = get_time_embedding(time, self.time_emb_dim)
        time_emb = self.time_proj(time_emb)

        down_outs = []
        for down in self.downs:
            out, skip = down(out, time_emb)
            down_outs.append(skip)

        for mid in self.mids:
            out = mid(out, time_emb)

        for up in self.ups:
            skip = down_outs.pop()
            out = up(out, skip, time_emb)

        out = self.norm_out(out)
        out = nn.SiLU()(out)
        return self.conv_out(out)
