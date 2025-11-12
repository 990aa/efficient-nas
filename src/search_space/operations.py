import torch
import torch.nn as nn


class SeparableConv2d(nn.Module):
    """Depthwise separable convolution for CPU efficiency."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x


class DilatedSeparableConv(nn.Module):
    """Dilated separable convolution for larger receptive fields."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 2
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv = SeparableConv2d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class GroupedConv2d(nn.Module):
    """Grouped convolution for better parameter efficiency."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        groups: int = 2,
        stride: int = 1,
    ):
        super().__init__()
        assert in_channels % groups == 0 and out_channels % groups == 0
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class InvertedBottleneck(nn.Module):
    """MobileNetV2 style inverted bottleneck block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: int = 6,
        kernel_size: int = 3,
    ):
        super().__init__()
        hidden_channels = in_channels * expansion
        self.use_residual = in_channels == out_channels

        layers = []
        # Expansion
        if expansion > 1:
            layers.extend(
                [
                    nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU6(inplace=True),
                ]
            )

        # Depthwise convolution
        padding = (kernel_size - 1) // 2
        layers.extend(
            [
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size,
                    padding=padding,
                    groups=hidden_channels,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU6(inplace=True),
            ]
        )

        # Projection
        layers.extend(
            [
                nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class SEInvertedBottleneck(nn.Module):
    """Inverted bottleneck with Squeeze-Excitation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: int = 6,
        kernel_size: int = 3,
    ):
        super().__init__()
        hidden_channels = in_channels * expansion
        self.use_residual = in_channels == out_channels

        # Expansion
        self.expand = (
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU6(inplace=True),
            )
            if expansion > 1
            else nn.Identity()
        )

        # Depthwise convolution
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size,
                padding=padding,
                groups=hidden_channels,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU6(inplace=True),
        )

        # SE attention
        self.se = SqueezeExcitation(hidden_channels)

        # Projection
        self.project = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.se(x)
        x = self.project(x)

        if self.use_residual:
            x += residual
        return x


# Operation pool definition
OPS = {
    "skip_connect": lambda C, stride: nn.Identity()
    if stride == 1
    else FactorizedReduce(C, C),
    "avg_pool_3x3": lambda C, stride: nn.AvgPool2d(3, stride=stride, padding=1),
    "max_pool_3x3": lambda C, stride: nn.MaxPool2d(3, stride=stride, padding=1),
    "sep_conv_3x3": lambda C, stride: SeparableConv2d(
        C, C, 3, stride=stride, padding=1
    ),
    "sep_conv_5x5": lambda C, stride: SeparableConv2d(
        C, C, 5, stride=stride, padding=2
    ),
    "sep_conv_7x7": lambda C, stride: SeparableConv2d(
        C, C, 7, stride=stride, padding=3
    ),
    "dil_sep_conv_3x3": lambda C, stride: DilatedSeparableConv(C, C, 3, dilation=2),
    "dil_sep_conv_5x5": lambda C, stride: DilatedSeparableConv(C, C, 5, dilation=2),
    "group_conv_2x2": lambda C, stride: GroupedConv2d(C, C, 3, groups=2, stride=stride),
    "group_conv_4x4": lambda C, stride: GroupedConv2d(C, C, 3, groups=4, stride=stride),
    "inv_bottleneck_2": lambda C, stride: InvertedBottleneck(C, C, expansion=2),
    "inv_bottleneck_4": lambda C, stride: InvertedBottleneck(C, C, expansion=4),
    "inv_bottleneck_6": lambda C, stride: InvertedBottleneck(C, C, expansion=6),
    "se_inv_bottleneck": lambda C, stride: SEInvertedBottleneck(C, C, expansion=4),
}


class FactorizedReduce(nn.Module):
    """Reduce feature map size without expensive convolution."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        assert out_channels % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, 1, stride=2, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels // 2, 1, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out
