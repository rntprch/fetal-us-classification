import warnings
from typing import Optional, Tuple, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torchvision.models import (
    resnet18, resnet34, ResNet18_Weights, ResNet34_Weights,
    densenet201, DenseNet201_Weights
)

from config import CLASSES

__all__ = [
    "ResNetDouble", "EffNetDouble", "EfficientResNet",
    "ResNet18", "DenseResNet", "EffNetB7"
]


def _suppress_warnings():
    """Context manager to ignore and restore warnings around interpolation."""
    return warnings.catch_warnings()


class ResNetDouble(nn.Module):
    """
    Ensemble of ResNet18 (shallow) and ResNet34 (detailed) backbones.

    Args:
        scale_shallow: Scale factor for shallow branch input.
        scale_detailed: Scale factor for detailed branch input.
        num_classes: Number of output classes.
        pretrained: Whether to load ImageNet-pretrained weights.
        finetune_backbone: If False, freezes backbone parameters.
    """

    def __init__(
        self,
        scale_shallow: float = 0.33,
        scale_detailed: float = 0.67,
        num_classes: int = len(CLASSES),
        pretrained: bool = True,
        finetune_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.scale_shallow = scale_shallow
        self.scale_detailed = scale_detailed
        self.num_classes = num_classes

        # Initialize shallow (ResNet18) and detailed (ResNet34) backbones
        self.backbone_shallow = nn.Sequential(
            *list(resnet18(pretrained=pretrained).children())[:-1],
            nn.Flatten()
        )
        self.backbone_detailed = nn.Sequential(
            *list(resnet34(pretrained=pretrained).children())[:-1],
            nn.Flatten()
        )
        self._maybe_freeze(finetune_backbone)

        # Combined classifier head
        combined_features = 512 + 512
        self.head = nn.Linear(combined_features, num_classes)
        if pretrained:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def _maybe_freeze(self, finetune: bool) -> None:
        """Freeze backbone parameters if finetuning is disabled."""
        if not finetune:
            for param in self.backbone_shallow.parameters():
                param.requires_grad = False
            for param in self.backbone_detailed.parameters():
                param.requires_grad = False

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract and concatenate features from both backbones."""
        with _suppress_warnings():
            warnings.simplefilter('ignore')
            xs = F.interpolate(
                x, scale_factor=self.scale_shallow,
                mode='bilinear', align_corners=True
            )
            xd = F.interpolate(
                x, scale_factor=self.scale_detailed,
                mode='bilinear', align_corners=True
            )
        feat_shallow = self.backbone_shallow(xs)
        feat_detailed = self.backbone_detailed(xd)
        return torch.cat([feat_shallow, feat_detailed], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))

    def forward_all(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (features, predictions)."""
        features = self.forward_features(x)
        return features, self.head(features)


class EffNetDouble(nn.Module):
    """
    Ensemble of EfficientNet-B0 (shallow) and B6 (detailed) models.

    Args:
        scale_shallow: Scale for shallow input (224x224).
        scale_detailed: Scale for detailed input (528x528).
        num_classes: Number of output classes.
        pretrained: Load pretrained weights if True.
        finetune_backbone: Freeze backbones if False.
    """

    def __init__(
        self,
        scale_shallow: float = 0.33,
        scale_detailed: float = 0.54,
        num_classes: int = len(CLASSES),
        pretrained: bool = True,
        finetune_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.scale_shallow = scale_shallow
        self.scale_detailed = scale_detailed
        self.num_classes = num_classes

        # Load EfficientNet backbones
        self.backbone_shallow = (
            EfficientNet.from_pretrained('efficientnet-b0')
            if pretrained else EfficientNet.from_name('efficientnet-b0')
        )
        self.n_features_shallow = self.backbone_shallow._fc.in_features
        self.backbone_shallow._fc = nn.Identity()

        self.backbone_detailed = (
            EfficientNet.from_pretrained('efficientnet-b6')
            if pretrained else EfficientNet.from_name('efficientnet-b6')
        )
        self.n_features_detailed = self.backbone_detailed._fc.in_features
        self.backbone_detailed._fc = nn.Identity()

        if not finetune_backbone:
            for param in self.backbone_shallow.parameters():
                param.requires_grad = False
            for param in self.backbone_detailed.parameters():
                param.requires_grad = False

        self.head = nn.Linear(in_features=self.n_features_shallow + self.n_features_detailed,
                              out_features=num_classes, bias=True)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Interpolate and extract pooled features."""
        with _suppress_warnings():
            warnings.simplefilter('ignore')
            xs = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)
            xd = F.interpolate(x, size=(528, 528), mode='bilinear', align_corners=True)
        # Extract features and apply global pooling
        fs = self.backbone_shallow.extract_features(xs)
        fd = self.backbone_detailed.extract_features(xd)
        fs = F.adaptive_avg_pool2d(fs, 1).view(fs.size(0), -1)
        fd = F.adaptive_avg_pool2d(fd, 1).view(fd.size(0), -1)
        return torch.cat([fs, fd], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))

    def forward_all(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.forward_features(x)
        return feats, self.head(feats)


class EfficientResNet(nn.Module):
    """
    Hybrid ensemble of ResNet34 and EfficientNet-B6.

    Args:
        size_shallow: Target size for ResNet34 input.
        size_detailed: Target size for EfficientNet input.
        num_classes: Number of output classes.
        pretrained: Load pretrained weights if True.
        finetune_backbone: Freeze backbones if False.
    """

    def __init__(
        self,
        size_shallow: Tuple[int, int] = (256, 256),
        size_detailed: Tuple[int, int] = (528, 528),
        num_classes: int = len(CLASSES),
        pretrained: bool = True,
        finetune_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.size_shallow = size_shallow
        self.size_detailed = size_detailed

        # ResNet34 shallow branch
        self.backbone_shallow = resnet34(
            weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else ResNet34_Weights.DEFAULT
        )
        self.n_features_shallow = self.backbone_shallow.fc.in_features
        self.backbone_shallow.fc = nn.Identity()

        # EfficientNet-B6 detailed branch
        self.backbone_detailed = (
            EfficientNet.from_pretrained('efficientnet-b6')
            if pretrained else EfficientNet.from_name('efficientnet-b6')
        )
        self.n_detailed_features = self.backbone_detailed._fc.in_features
        self.backbone_detailed._fc = nn.Identity()

        if not finetune_backbone:
            for p in self.backbone_shallow.parameters(): p.requires_grad = False
            for p in self.backbone_detailed.parameters(): p.requires_grad = False

        self.head = nn.Linear(self.n_features_shallow + self.n_detailed_features, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        with _suppress_warnings():
            warnings.simplefilter('ignore')
            xs = F.interpolate(x, size=self.size_shallow, mode='bilinear', align_corners=True)
            xd = F.interpolate(x, size=self.size_detailed, mode='bilinear', align_corners=True)
        fs = self.backbone_shallow(xs)
        fd = self.backbone_detailed.extract_features(xd)
        fd = F.adaptive_avg_pool2d(fd, 1).view(fd.size(0), -1)
        return torch.cat([fs, fd], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))

    def forward_all(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.forward_features(x)
        return feats, self.head(feats)


class ResNet18(nn.Module):
    """
    Single ResNet18 with optional input scaling.

    Args:
        scale: Scale factor for input interpolation.
        num_classes: Number of output classes.
        pretrained: Load pretrained weights.
        finetune_backbone: Freeze backbone if False.
    """

    def __init__(
        self,
        scale: float = 0.33,
        num_classes: int = len(CLASSES),
        pretrained: bool = True,
        finetune_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.backbone = nn.Sequential(
            *list(resnet18(pretrained=pretrained).children())[:-1],
            nn.Flatten()
        )
        if not finetune_backbone:
            for p in self.backbone.parameters(): p.requires_grad = False
        self.head = nn.Linear(512, num_classes)
        if pretrained:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        with _suppress_warnings():
            warnings.simplefilter('ignore')
            x_scaled = x if self.scale == 1.0 else F.interpolate(
                x, scale_factor=self.scale,
                mode='bilinear', align_corners=True
            )
        return self.backbone(x_scaled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))

    def forward_all(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.forward_features(x)
        return feats, self.head(feats)


class DenseResNet(nn.Module):
    """
    Ensemble of ResNet18 (shallow) and DenseNet201 (detailed) with optional gating.

    Args:
        scale_shallow: Scale factor for shallow branch.
        scale_detailed: Scale factor for detailed branch.
        num_classes: Number of output classes.
        pretrained: Load pretrained weights.
        finetune_backbone: Freeze backbones if False.
        gating: Fuse features weighted by a learnable alpha.
    """

    def __init__(
        self,
        scale_shallow: float = 0.33,
        scale_detailed: float = 0.54,
        num_classes: int = len(CLASSES),
        pretrained: bool = True,
        finetune_backbone: bool = True,
        gating: bool = False,
    ) -> None:
        super().__init__()
        self.scale_shallow = scale_shallow
        self.scale_detailed = scale_detailed
        self.gating = gating

        # Shallow ResNet18 branch
        self.backbone_shallow = resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else ResNet18_Weights.DEFAULT
        )
        self.n_shallow_features = self.backbone_shallow.fc.in_features
        self.backbone_shallow.fc = nn.Identity()

        # Detailed DenseNet201 branch
        self.backbone_detailed = densenet201(
            weights=DenseNet201_Weights.IMAGENET1K_V1 if pretrained else DenseNet201_Weights.DEFAULT
        )
        self.n_detailed_features = self.backbone_detailed.classifier.in_features
        self.backbone_detailed.classifier = nn.Identity()

        if not finetune_backbone:
            for p in self.backbone_shallow.parameters(): p.requires_grad = False
            for p in self.backbone_detailed.parameters(): p.requires_grad = False

        self.head = nn.Linear(self.n_shallow_features + self.n_detailed_features, num_classes)
        # Learnable gating parameter
        self.alpha = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with _suppress_warnings():
            warnings.simplefilter('ignore')
            xs = F.interpolate(x, scale_factor=self.scale_shallow,
                                mode='bilinear', align_corners=True)
            xd = F.interpolate(x, scale_factor=self.scale_detailed,
                                mode='bilinear', align_corners=True)
        fs = self.backbone_shallow(xs)
        fd = self.backbone_detailed(xd)
        if self.gating:
            fs = self.alpha * fs
            fd = (1 - self.alpha) * fd
        fused = torch.cat([fs, fd], dim=1)
        return self.head(fused)


class EffNetB7(nn.Module):
    """
    EfficientNet-B7 classifier with resized input.

    Args:
        pretrained: Load pretrained weights if True.
        num_classes: Number of output classes.
    """

    def __init__(self, pretrained: bool = True, num_classes: int = len(CLASSES)) -> None:
        super().__init__()
        self.backbone = (
            EfficientNet.from_pretrained('efficientnet-b7') if pretrained
            else EfficientNet.from_name('efficientnet-b7')
        )
        # Replace final FC layer
        in_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # EfficientNet-B7 expects at least 600x600 input
        x_resized = F.interpolate(x, size=(600, 600), mode='bilinear', align_corners=True)
        return self.backbone(x_resized)