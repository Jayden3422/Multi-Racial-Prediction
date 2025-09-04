from typing import Optional, Any, Dict
import torch
import torch.nn as nn
from torchvision.models import (
    resnet18 as tv_resnet18,
    resnet34 as tv_resnet34,
    resnet50 as tv_resnet50,
    resnet101 as tv_resnet101,
    resnet152 as tv_resnet152,
)

def _strip_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k[7:]
        new_sd[k] = v
    return new_sd

def _load_flexible(model: nn.Module, pretrain_pth: str):
    sd = torch.load(pretrain_pth, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if not isinstance(sd, dict):
        raise ValueError("Invalid state_dict in pretrain_pth")
    sd = _strip_prefix(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[pretrain] loaded with missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing keys sample:", missing[:5])
    if unexpected:
        print("  unexpected keys sample:", unexpected[:5])

def _ensure_head(model: nn.Module, num_classes: int):
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_f = model.fc.in_features
        model.fc = nn.Linear(in_f, num_classes)

def _build(builder, *, pretrain_pth: Optional[str], num_classes: int, **kwargs: Any):
    m = builder(weights=None, **kwargs)
    _ensure_head(m, num_classes)
    if pretrain_pth:
        _load_flexible(m, pretrain_pth)
        _ensure_head(m, num_classes)
    return m

def resnet18(*, pretrain_pth: Optional[str] = None, num_classes: int = 1000, **kwargs: Any):
    return _build(tv_resnet18, pretrain_pth=pretrain_pth, num_classes=num_classes, **kwargs)

def resnet34(*, pretrain_pth: Optional[str] = None, num_classes: int = 1000, **kwargs: Any):
    return _build(tv_resnet34, pretrain_pth=pretrain_pth, num_classes=num_classes, **kwargs)

def resnet50(*, pretrain_pth: Optional[str] = None, num_classes: int = 1000, **kwargs: Any):
    return _build(tv_resnet50, pretrain_pth=pretrain_pth, num_classes=num_classes, **kwargs)

def resnet101(*, pretrain_pth: Optional[str] = None, num_classes: int = 1000, **kwargs: Any):
    return _build(tv_resnet101, pretrain_pth=pretrain_pth, num_classes=num_classes, **kwargs)

def resnet152(*, pretrain_pth: Optional[str] = None, num_classes: int = 1000, **kwargs: Any):
    return _build(tv_resnet152, pretrain_pth=pretrain_pth, num_classes=num_classes, **kwargs)
