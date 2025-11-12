import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class ResNet20(nn.Module):
    """ResNet-20 baseline for CIFAR-10."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Three stages with [3, 3, 3] blocks
        self.layer1 = self._make_layer(16, 3, stride=1)
        self.layer2 = self._make_layer(32, 3, stride=2)
        self.layer3 = self._make_layer(64, 3, stride=2)

        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class BasicBlock(nn.Module):
    """Basic ResNet block."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MobileNetV2(nn.Module):
    """MobileNetV2 baseline optimized for CPU."""

    def __init__(self, num_classes: int = 10, width_mult: float = 1.0):
        super().__init__()

        # Initial conv layer
        self.features = [
            nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU6(inplace=True),
            )
        ]

        # Inverted residual blocks
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1],  # Changed stride from 2 to 1 for CIFAR
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        input_channel = 32
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(
                    InvertedResidual(
                        input_channel, output_channel, stride, expand_ratio=t
                    )
                )
                input_channel = output_channel

        self.features.append(
            nn.Sequential(
                nn.Conv2d(input_channel, 1280, 1, bias=False),
                nn.BatchNorm2d(1280),
                nn.ReLU6(inplace=True),
            )
        )

        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.mean([2, 3])  # Global average pooling
        x = self.classifier(x)
        return x


class InvertedResidual(nn.Module):
    """Inverted residual block for MobileNetV2."""

    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(inp, hidden_dim, 1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                )
            )

        layers.extend(
            [
                nn.Conv2d(
                    hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, bias=False),
                nn.BatchNorm2d(oup),
            ]
        )

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ShuffleNet(nn.Module):
    """ShuffleNet baseline for efficient CPU inference."""

    def __init__(self, num_classes: int = 10, groups: int = 2):
        super().__init__()
        self.groups = groups

        # Initial layers
        self.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ShuffleNet stages
        self.stage2 = self._make_stage(24, 144, 3, 2)
        self.stage3 = self._make_stage(144, 288, 7, 2)
        self.stage4 = self._make_stage(288, 576, 3, 2)

        self.conv5 = nn.Conv2d(576, 1024, kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm2d(1024)

        self.fc = nn.Linear(1024, num_classes)

    def _make_stage(
        self, inp: int, oup: int, repetitions: int, stride: int
    ) -> nn.Sequential:
        layers = [ShuffleUnit(inp, oup, stride, self.groups)]
        for _ in range(repetitions - 1):
            layers.append(ShuffleUnit(oup, oup, 1, self.groups))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ShuffleUnit(nn.Module):
    """ShuffleNet unit with channel shuffle."""

    def __init__(self, inp: int, oup: int, stride: int, groups: int):
        super().__init__()
        self.stride = stride

        mid_channels = oup // 4

        # Branch 1
        self.branch1 = nn.Sequential()
        if stride == 2:
            self.branch1 = nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, mid_channels, 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
            )

        # Branch 2
        self.branch2 = nn.Sequential(
            nn.Conv2d(
                inp if stride == 1 else mid_channels, mid_channels, 1, bias=False
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels, mid_channels, 3, stride, 1, groups=groups, bias=False
            ),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, oup - inp if stride == 2 else oup, 1, bias=False),
            nn.BatchNorm2d(oup - inp if stride == 2 else oup),
        )

        if stride == 2:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, inp, 3, 2, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, oup - mid_channels, 1, bias=False),
                nn.BatchNorm2d(oup - mid_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = self.channel_shuffle(out, 2)
        return out

    def channel_shuffle(self, x: torch.Tensor, groups: int) -> torch.Tensor:
        batch_size, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups

        # Reshape
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batch_size, -1, height, width)

        return x


class BaselineManager:
    """Manages baseline model evaluation and comparison."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.baselines = {
            "resnet20": ResNet20,
            "mobilenetv2": MobileNetV2,
            "shufflenet": ShuffleNet,
        }

    def evaluate_baseline(
        self, model_name: str, dataset: str = "cifar10"
    ) -> Dict[str, Any]:
        """Evaluate a baseline model."""
        if model_name not in self.baselines:
            raise ValueError(f"Unknown baseline: {model_name}")

        from ..utils.data_loader import DatasetManager
        from ..evaluation.training import TrainingProtocol

        dataset_manager = DatasetManager()
        training_protocol = TrainingProtocol(self.device)

        # Get data loaders
        if dataset == "cifar10":
            train_loader, _, test_loader = dataset_manager.get_cifar10_loaders(
                cutout=True
            )
        elif dataset == "cifar100":
            train_loader, test_loader = dataset_manager.get_cifar100_loaders(
                cutout=True
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        # Create model
        num_classes = dataset_manager.dataset_info[dataset]["num_classes"]
        model_class = self.baselines[model_name]
        model = model_class(num_classes=num_classes)

        # Evaluate
        results = training_protocol.final_evaluation(model, train_loader, test_loader)
        results["model_name"] = model_name
        results["dataset"] = dataset

        # Add hardware metrics
        from ..evaluation.zero_cost_proxies import ZeroCostProxies

        zc = ZeroCostProxies(self.device)
        x, y = next(iter(test_loader))
        x, y = x.to(self.device), y.to(self.device)

        results["latency"] = zc.estimate_latency(model, x)
        results["params"] = zc.parameter_count(model)
        results["flops"] = zc.estimate_flops(model, x)

        return results

    def evaluate_all_baselines(self, dataset: str = "cifar10") -> Dict[str, Any]:
        """Evaluate all baseline models."""
        results = {}
        for model_name in self.baselines.keys():
            print(f"Evaluating {model_name} on {dataset}...")
            results[model_name] = self.evaluate_baseline(model_name, dataset)

        return results
