import torch.nn as nn

from ..spec import BackboneSpec
from .edgeconv import EDGECONV_BACKBONE_SPEC
from .dp3 import DP3_BACKBONE_SPEC
from .handcrafted import HANDCRAFTED_BACKBONE_SPEC
from .pointnet import POINTNET_BACKBONE_SPEC
from .pointnetplusplus import POINTNETPLUSPLUS_BACKBONE_SPEC


POINT_BACKBONE_SPECS: dict[str, BackboneSpec] = {
    "pointnet": POINTNET_BACKBONE_SPEC,
    "edgeconv": EDGECONV_BACKBONE_SPEC,
    "pointnetplusplus": POINTNETPLUSPLUS_BACKBONE_SPEC,
    "handcrafted": HANDCRAFTED_BACKBONE_SPEC,
    "dp3": DP3_BACKBONE_SPEC,
}


def build_point_backbone(backbone: str, **kwargs) -> nn.Module:
    key = str(backbone).lower()
    try:
        return POINT_BACKBONE_SPECS[key].build(**kwargs)
    except KeyError as exc:
        raise ValueError(
            f"Unsupported backbone '{backbone}'. Choose from: {', '.join(sorted(POINT_BACKBONE_SPECS))}."
        ) from exc
