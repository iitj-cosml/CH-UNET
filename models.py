import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import R2Score
from losses import conservation_loss
from utilities import enforce_conservation

# =====================================================
# Periodic Convolution
# =====================================================
class PeriodicConv2d(nn.Module):
    """
    Conv2d with circular (periodic) padding.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=True):
        super().__init__()
        self.stride = stride
        self.pad = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=0, bias=bias)

    def forward(self, x):
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode="circular")
        return self.conv(x)


# =====================================================
# Residual Block
# =====================================================
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, n_groups=8, activation="gelu"):
        super().__init__()
        act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.norm1 = nn.GroupNorm(num_groups=n_groups, num_channels=in_ch)
        self.conv1 = PeriodicConv2d(in_ch, out_ch, kernel_size=3)
        self.norm2 = nn.GroupNorm(num_groups=n_groups, num_channels=out_ch)
        self.conv2 = PeriodicConv2d(out_ch, out_ch, kernel_size=3)
        self.act = act
        self.residual = (
            PeriodicConv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.residual(x)


# =====================================================
# Attention Block
# =====================================================
class AttentionBlock(nn.Module):
    def __init__(self, channels, n_heads=4, n_groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=n_groups, num_channels=channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.n_heads = n_heads

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W)
        qkv = self.qkv(h)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        q = q.view(B, self.n_heads, C // self.n_heads, H * W)
        k = k.view(B, self.n_heads, C // self.n_heads, H * W)
        v = v.view(B, self.n_heads, C // self.n_heads, H * W)

        scale = (C // self.n_heads) ** -0.5
        attn = torch.einsum("bhcl,bhcm->bhlm", q * scale, k)
        attn = torch.softmax(attn, dim=-1)

        h = torch.einsum("bhlm,bhcm->bhcl", attn, v).reshape(B, C, H * W)
        h = self.proj(h).view(B, C, H, W)
        return x + h


# =====================================================
# Down / Up Blocks
# =====================================================
class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, n_blocks=2, n_groups=8, use_attn=False):
        super().__init__()
        blocks = []
        for i in range(n_blocks):
            blocks.append(ResidualBlock(in_ch if i == 0 else out_ch, out_ch, n_groups))
            if use_attn:
                blocks.append(AttentionBlock(out_ch, n_heads=4, n_groups=n_groups))
        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, n_blocks=2, n_groups=8, use_attn=False):
        super().__init__()
        blocks = []
        for i in range(n_blocks):
            blocks.append(ResidualBlock(in_ch if i == 0 else out_ch, out_ch, n_groups))
            if use_attn:
                blocks.append(AttentionBlock(out_ch, n_heads=4, n_groups=n_groups))
        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = PeriodicConv2d(ch, ch, kernel_size=3, stride=2)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.op = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


# =====================================================
# Middle Block
# =====================================================
class MiddleBlock(nn.Module):
    def __init__(self, ch, n_groups=8, use_attn=True):
        super().__init__()
        self.block1 = ResidualBlock(ch, ch, n_groups)
        self.attn = AttentionBlock(ch, n_heads=4, n_groups=n_groups) if use_attn else nn.Identity()
        self.block2 = ResidualBlock(ch, ch, n_groups)

    def forward(self, x):
        return self.block2(self.attn(self.block1(x)))


# =====================================================
# U-Net
# =====================================================
class UNet(nn.Module):
    def __init__(
        self,
        in_ch=1,
        out_ch=1,
        hidden_ch=64,
        ch_mults=(1, 2, 4, 8),
        n_blocks=2,
        n_groups=8,
        attn_resolutions=(16,),  # add attention at these resolutions
        use_attn_mb=True
    ):
        super().__init__()

        self.input_proj = PeriodicConv2d(in_ch, hidden_ch, kernel_size=3)

        # Encoder
        self.downs = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.use_attn_mb = use_attn_mb
        in_channels = hidden_ch
        self.skip_channels = []
        for i, mult in enumerate(ch_mults):
            out_channels = hidden_ch * mult
            use_attn = (2 ** i) in attn_resolutions
            self.downs.append(DownBlock(in_channels, out_channels, n_blocks, n_groups, use_attn))
            in_channels = out_channels
            self.skip_channels.append(out_channels)
            if i != len(ch_mults) - 1:
                self.downsamples.append(Downsample(in_channels))

        # Middle
        self.mid = MiddleBlock(in_channels, n_groups, use_attn=self.use_attn_mb)

        # Decoder
        self.ups = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mults))):
            out_channels = hidden_ch * mult
            use_attn = (2 ** i) in attn_resolutions
            self.ups.append(UpBlock(in_channels + out_channels, out_channels, n_blocks, n_groups, use_attn))
            in_channels = out_channels
            if i != 0:
                self.upsamples.append(Upsample(in_channels, in_channels))

        # Output
        self.out_norm = nn.GroupNorm(num_groups=n_groups, num_channels=in_channels)
        self.out_act = nn.GELU()
        self.out_conv = PeriodicConv2d(in_channels, out_ch, kernel_size=3)

    def forward(self, x):
        x = self.input_proj(x)

        skips = []
        for i, down in enumerate(self.downs):
            x = down(x)
            skips.append(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)

        x = self.mid(x)

        for i, up in enumerate(self.ups):
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = up(x)
            if i < len(self.upsamples):
                x = self.upsamples[i](x)

        x = self.out_conv(self.out_act(self.out_norm(x)))
        return x


# ---------------------------
# Lightning Module
# ---------------------------
class LitUNet(pl.LightningModule):
    def __init__(self, lr=1e-4, alpha=0, enf_cons=True, use_attn=False, dtype: torch.dtype = torch.float64):
        super().__init__()
        self.save_hyperparameters(ignore=['dtype'])
        self.hparams['dtype'] = 'float32' if dtype == torch.float32 else 'float64'
        self.alpha = alpha
        self.enf_cons = bool(enf_cons)
        self.lr = lr
        self.use_attn = use_attn
        self.train_r2 = R2Score()
        self.val_r2 = R2Score()

        self.model = UNet(in_ch=1, out_ch=1, hidden_ch=64, ch_mults=(1,2,4,8), n_blocks=2, use_attn_mb=self.use_attn)

        if dtype == torch.float32:
            self.model.float()
        else:
            self.model.double()

    def forward(self, x):
        return self.model(x)

    def _param_dtype(self):
        return next(self.model.parameters()).dtype

    def training_step(self, batch, batch_idx):
        x, y = batch
        param_dtype = self._param_dtype()
        if x.dtype != param_dtype:
            x = x.to(dtype=param_dtype)
        if y.dtype != param_dtype:
            y = y.to(dtype=param_dtype)

        y_hat = self(x)

        if self.enf_cons:
            y_hat = enforce_conservation(y_hat, x)

        loss = conservation_loss(y_hat, y, alpha=self.alpha)

        # flatten for metrics
        y_hat_flat = y_hat.view(y_hat.size(0), -1)
        y_flat = y.view(y.size(0), -1)

        self.train_r2.update(y_hat_flat, y_flat)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        param_dtype = self._param_dtype()
        if x.dtype != param_dtype:
            x = x.to(dtype=param_dtype)
        if y.dtype != param_dtype:
            y = y.to(dtype=param_dtype)

        y_hat = self(x)

        if self.enf_cons:
            y_hat = enforce_conservation(y_hat, x)

        loss = conservation_loss(y_hat, y, alpha=self.alpha)

        # flatten for metrics
        y_hat_flat = y_hat.view(y_hat.size(0), -1)
        y_flat = y.view(y.size(0), -1)

        self.val_r2.update(y_hat_flat, y_flat)

        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train_r2", self.train_r2.compute(), prog_bar=True, on_epoch=True)
        self.train_r2.reset()

    def on_validation_epoch_end(self):
        self.log("val_r2", self.val_r2.compute(), prog_bar=True, on_epoch=True)
        self.val_r2.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
