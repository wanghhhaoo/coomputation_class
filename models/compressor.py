"""
models/compressor.py — 特征压缩/解压 (v5.2)

版本历史：
  v3: (B, 128, 32, 32) → 压缩 (B, 32, 4, 4) → 解压 (B, 64, 8, 8)   256× 压缩比
  v5.1: (B, 24, 8, 8) → 压缩 (B, 24, 4, 4) → 解压 (B, 48, 8, 8)   4× 压缩比
        in_channels == out_channels 时跳过 1×1 Conv，只做空间池化
  v5.2: spatial_size=None → 无空间压缩（Phase A 验证上界）
        pool = nn.Identity()，特征原样透传
"""

import torch
import torch.nn as nn


class FeatureCompressor(nn.Module):
    """
    v5.2:
      spatial_size=None  → Phase A 无压缩（Identity pool，直接透传）
      spatial_size=14    → 28→14，4× 空间压缩
      spatial_size=7     → 28→7，16× 空间压缩
      in_channels == out_channels 时跳过 1×1 Conv
    """
    def __init__(
        self,
        in_channels: int  = 24,
        out_channels: int = 24,
        spatial_size      = None,   # None = 不压缩（Phase A）
        quantize: bool    = False,
    ):
        super().__init__()
        self.quantize     = quantize
        self.spatial_size = spatial_size

        # 只在通道数不同时才加 1×1 Conv
        if in_channels != out_channels:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv = nn.Identity()

        # v5.2: spatial_size=None → Identity（无空间压缩）
        if spatial_size is None:
            self.pool = nn.Identity()
        else:
            self.pool = nn.AdaptiveAvgPool2d(spatial_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x)
        if self.quantize:
            x = self._int8_quantize(x)
        return x

    @staticmethod
    def _int8_quantize(x: torch.Tensor) -> torch.Tensor:
        scale = x.abs().max() / 127.0
        x_q   = (x / (scale + 1e-8)).clamp(-128, 127).to(torch.int8)
        return x_q.float() * scale

    def compress_info(self) -> str:
        if self.spatial_size is None:
            return "压缩: 无（Phase A 无压缩模式）"
        orig = 24 * 28 * 28
        comp = 24 * self.spatial_size * self.spatial_size
        ratio = orig / max(comp, 1)
        return f"压缩: {orig:,} → {comp:,}  ({ratio:.1f}× 压缩比)"


class FeatureDecompressor(nn.Module):
    """
    v5.1: in (B, 24, 4, 4) → out (B, 48, 8, 8)
    通道扩展（24→48），空间上采样（4→8）
    """
    def __init__(
        self,
        in_channels: int  = 24,
        out_channels: int = 48,
        target_size: int  = 8,
    ):
        super().__init__()
        mid = in_channels * 2   # 24→48

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.upsample = nn.Upsample(
            size=target_size, mode='bilinear', align_corners=False
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.upsample(x)
        x = self.conv2(x)
        return x


class CompressorPipeline(nn.Module):
    """压缩 + 解压完整 pipeline（单设备模式）"""
    def __init__(self, quantize: bool = False):
        super().__init__()
        self.compressor   = FeatureCompressor(quantize=quantize)
        self.decompressor = FeatureDecompressor()

    def forward(self, x):
        return self.decompressor(self.compressor(x))

    def compress_only(self, x):
        return self.compressor(x)

    def decompress_only(self, x):
        return self.decompressor(x)
