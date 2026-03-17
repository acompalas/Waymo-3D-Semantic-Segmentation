import torch


POINT_FEATURE_DIM = 2
RANGE_IMAGE_CHANNELS = 4


def point_input_dim(*, geometry_only: bool, point_feature_dim: int = POINT_FEATURE_DIM) -> int:
    return 3 if bool(geometry_only) else 3 + int(point_feature_dim)


def range_input_channels(*, geometry_only: bool) -> int:
    return 1 if bool(geometry_only) else RANGE_IMAGE_CHANNELS


def select_point_model_inputs(
    points: torch.Tensor,
    point_features: torch.Tensor,
    *,
    geometry_only: bool,
) -> torch.Tensor:
    points = points.float()
    point_features = point_features.float()
    if bool(geometry_only):
        return points
    return torch.cat([points, point_features], dim=-1)


def point_geometry(model_inputs: torch.Tensor) -> torch.Tensor:
    return model_inputs[..., :3].float()


def select_range_model_inputs(range_images: torch.Tensor, *, geometry_only: bool) -> torch.Tensor:
    range_images = range_images.float()
    if bool(geometry_only):
        return range_images[..., :1]
    return range_images
