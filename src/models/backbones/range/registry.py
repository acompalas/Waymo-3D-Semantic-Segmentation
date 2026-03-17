import torch.nn as nn

from ..spec import BackboneSpec
from .crossattn_unet import CROSSATTN_UNET_BACKBONE_SPEC
from .unet import UNET_BACKBONE_SPEC


RANGE_BACKBONE_SPECS: dict[str, BackboneSpec] = {
    "unet": UNET_BACKBONE_SPEC,
    "crossattn_unet": CROSSATTN_UNET_BACKBONE_SPEC,
}


def build_range_backbone(backbone: str, **kwargs) -> nn.Module:
    key = str(backbone).lower()
    try:
        return RANGE_BACKBONE_SPECS[key].build(**kwargs)
    except KeyError as exc:
        raise ValueError(
            f"Unsupported backbone '{backbone}'. Choose from: {', '.join(sorted(RANGE_BACKBONE_SPECS))}."
        ) from exc
