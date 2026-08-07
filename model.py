import torch
import torch.nn as nn

def get_time_embedding(time_steps, time_emb_dim):
    half_dim = time_emb_dim // 2
    factor = torch.exp(
        -torch.log(torch.tensor(10000.0)) * torch.arange(half_dim, device=time_steps.device) / half_dim
    )
    time_embed = time_steps[:, None] * factor[None, :]
    time_embed = torch.cat([torch.sin(time_embed), torch.cos(time_embed)], dim=-1)
    return time_embed

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, down_sample, num_heads):
        super().__init__()
        self.down_sample = down_sample
        self.resnet_conv_first = nn.Sequential(
            nn.GroupNorm(num_groups = 8, num_channels = in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1, stride = 1)
        )
        self.time_emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.resnet_conv_second = nn.Sequential(
            nn.GroupNorm(num_groups = 8, num_channels = out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1, stride = 1)
        )
        self.attention_norm = nn.GroupNorm(num_groups = 8, num_channels = out_channels)
        self.attention = nn.MultiheadAttention(embed_dim = out_channels, num_heads = num_heads, batch_first = True)
        self.residual_input_conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1)
        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 2, padding = 1) if self.down_sample else nn.Identity()


    def forward(self, x, time_emb):
        out = x

        # First ResNet block
        resnet_input = out
        out = self.resnet_conv_first(out)
        out = out + self.time_emb_layers(time_emb)[:, :, None, None]
        out = self.resnet_conv_second(out)
        out = out + self.residual_input_conv(resnet_input)

        # Attention block
        batch_size, channels, height, width = out.shape
        in_attn = out.reshape(batch_size, channels, height * width)
        in_attn = self.attention_norm(in_attn)
        in_attn = in_attn.transpose(1, 2)
        out_attn = self.attention(in_attn, in_attn, in_attn, need_weights=False)[0]
        out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, height, width)
        out = out + out_attn

        skip = out
        out = self.down_sample_conv(out)
        return out, skip

class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, num_heads):
        super().__init__()
        self.resnet_conv_first = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups = 8, num_channels = in_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1, stride = 1)
            ),
            nn.Sequential(
                nn.GroupNorm(num_groups = 8, num_channels = in_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1, stride = 1)
            )
        ])
        self.time_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels)
            ),
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels)
            )
        ])
        self.resnet_conv_second = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups = 8, num_channels = out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1, stride = 1)
            ),
            nn.Sequential(
                nn.GroupNorm(num_groups = 8, num_channels = out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1, stride = 1)
            )
        ])
        self.attention_norm = nn.GroupNorm(num_groups = 8, num_channels = out_channels)
        self.attention = nn.MultiheadAttention(embed_dim = out_channels, num_heads = num_heads, batch_first = True)
        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size = 1),
            nn.Conv2d(in_channels, out_channels, kernel_size = 1)
        ])

    def forward(self, x, time_emb):
        out = x
        # First ResNet block
        resnet_input = out
        out = self.resnet_conv_first[0](out)
        out = out + self.time_emb_layers[0](time_emb)[:, :, None, None]
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)
        
        # Attention block
        batch_size, channels, height, width = out.shape
        in_attn = out.reshape(batch_size, channels, height * width)
        in_attn = self.attention_norm(in_attn)
        in_attn = in_attn.transpose(1, 2)
        out_attn = self.attention(in_attn, in_attn, in_attn, need_weights=False)[0]
        out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, height, width)
        out = out + out_attn

        # Second ResNet block
        resnet_input = out
        out = self.resnet_conv_first[1](out)
        out = out + self.time_emb_layers[1](time_emb)[:, :, None, None]
        out = self.resnet_conv_second[1](out)
        out = out + self.residual_input_conv[1](resnet_input)

        return out

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, up_sample, num_heads):
        super().__init__()
        self.up_sample = up_sample
        self.resnet_conv_first = nn.Sequential(
            nn.GroupNorm(num_groups = 8, num_channels = in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1, stride = 1)
        )
        self.time_emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.resnet_conv_second = nn.Sequential(
            nn.GroupNorm(num_groups = 8, num_channels = out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1, stride = 1)
        )

        self.attention_norm = nn.GroupNorm(num_groups = 8, num_channels = out_channels)
        self.attention = nn.MultiheadAttention(embed_dim = out_channels, num_heads = num_heads, batch_first = True)
        self.residual_input_conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1)
        self.up_sample_conv = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size = 4, stride = 2, padding = 1) if self.up_sample else nn.Identity()

    def forward(self, x, out_down, time_emb):

        x = self.up_sample_conv(x)
        x = torch.cat([x, out_down], dim = 1)

        out = x

        # First ResNet block
        resnet_input = out
        out = self.resnet_conv_first(out)
        out = out + self.time_emb_layers(time_emb)[:, :, None, None]
        out = self.resnet_conv_second(out)
        out = out + self.residual_input_conv(resnet_input)

        # Attention block
        batch_size, channels, height, width = out.shape
        in_attn = out.reshape(batch_size, channels, height * width)
        in_attn = self.attention_norm(in_attn)
        in_attn = in_attn.transpose(1, 2)
        out_attn = self.attention(in_attn, in_attn, in_attn, need_weights=False)[0]
        out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, height, width)
        out = out + out_attn

        return out

class Unet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.down_channels = [32, 64, 128, 256]
        self.mid_channels = [256, 256, 256]
        self.time_emb_dim = 128
        self.down_sample = [True, True, False]

        self.time_proj = nn.Sequential(
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
            nn.SiLU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim)
        )
        self.up_sample = list(reversed(self.down_sample))
        self.conv_in = nn.Conv2d(in_channels, self.down_channels[0], kernel_size = 3, padding = 1)

        self.downs = nn.ModuleList([])
        for i in range (len(self.down_channels) - 1):
            self.downs.append(DownBlock(self.down_channels[i], self.down_channels[i + 1], self.time_emb_dim, self.down_sample[i], num_heads = 4))

        self.mids = nn.ModuleList([])
        for i in range (len(self.mid_channels) - 1):
            self.mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i + 1], self.time_emb_dim, num_heads = 4))

        self.ups = nn.ModuleList([])
        for i in reversed(range(len(self.down_channels) - 1)):
            self.ups.append(UpBlock(self.down_channels[i + 1] * 2, self.down_channels[i] if i > 0 else 16, self.time_emb_dim, self.down_sample[i], num_heads = 4))

        self.norm_out = nn.GroupNorm(num_groups = 8, num_channels = 16)
        self.conv_out = nn.Conv2d(16, in_channels, kernel_size = 3, padding = 1)

    def forward(self, x, time):
        out = self.conv_in(x)
        time_emb = get_time_embedding(time, self.time_emb_dim).to(x.device)
        time_emb = self.time_proj(time_emb)

        down_outs = []
        for down in self.downs:
            out, skip = down(out, time_emb)
            down_outs.append(skip)

        for mid in self.mids:
            out = mid(out, time_emb)

        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, time_emb)

        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        return out