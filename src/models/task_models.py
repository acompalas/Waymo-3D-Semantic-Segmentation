import torch

from .base import PointCloudSegmentationModel, RangeImageSegmentationModel
from .backbones.point.registry import build_point_backbone
from .backbones.range.registry import build_range_backbone
from .behaviors import build_point_behavior, build_range_behavior
from .heads import PointMLPHead, RangeConvHead
from .inputs import point_geometry, point_input_dim, range_input_channels, select_point_model_inputs, select_range_model_inputs


def _build_point_head(head: str, *, input_dim: int, output_dim: int) -> PointMLPHead:
    key = str(head).lower()
    if key == "mlp":
        return PointMLPHead(input_dim=input_dim, output_dim=output_dim)
    raise ValueError(f"Unsupported point head '{head}'.")


def _build_range_head(head: str, *, input_channels: int, output_channels: int) -> RangeConvHead:
    key = str(head).lower()
    if key == "segmentation":
        return RangeConvHead(input_channels=input_channels, output_channels=output_channels, kernel_size=1)
    if key == "denoising":
        return RangeConvHead(input_channels=input_channels, output_channels=output_channels, kernel_size=3)
    raise ValueError(f"Unsupported range head '{head}'.")

class PointCloudTaskModel(PointCloudSegmentationModel):
    def __init__(
        self,
        num_classes: int = 23,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        backbone: str = "edgeconv",
        head: str = "mlp",
        behavior: str = "supervised",
        hidden_dim: int = 256,
        depth: int = 6,
        knn_k: int = 16,
        dropout: float = 0.1,
        diffusion_steps: int = 1000,
        use_balanced_class_weights: bool = True,
        geometry_only: bool = False,
        knn_scales: tuple[int, ...] = (16, 32, 64),
        knn_support_size: int = 16384,
        knn_query_chunk: int = 4096,
        proj_dim: int = 0,
        proj_depth: int = 0,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            use_balanced_class_weights=use_balanced_class_weights,
        )
        self.save_hyperparameters()
        self.behavior_impl = build_point_behavior(behavior, diffusion_steps=int(diffusion_steps))
        model_input_dim = point_input_dim(geometry_only=bool(geometry_only))
        backbone_input_dim = self.behavior_impl.point_backbone_input_dim(
            point_input_dim=model_input_dim,
            num_classes=int(num_classes),
        )
        self.backbone = build_point_backbone(
            str(backbone),
            input_dim=backbone_input_dim,
            hidden_dim=int(hidden_dim),
            depth=int(depth),
            dropout=float(dropout),
            knn_k=int(knn_k),
            time_dim=self.behavior_impl.time_dim,
            knn_scales=tuple(int(k) for k in knn_scales),
            knn_support_size=int(knn_support_size),
            knn_query_chunk=int(knn_query_chunk),
            proj_dim=int(proj_dim),
            proj_depth=int(proj_depth),
            proj_dropout=float(proj_dropout),
        )
        self.head = _build_point_head(str(head), input_dim=int(self.backbone.output_dim), output_dim=int(num_classes))

    def prepare_runtime(self) -> None:
        self.behavior_impl.prepare_runtime(self)

    def point_model_inputs(self, points: torch.Tensor, point_features: torch.Tensor) -> torch.Tensor:
        return select_point_model_inputs(
            points,
            point_features,
            geometry_only=bool(self.hparams.geometry_only),
        )

    def predict_logits(self, points: torch.Tensor, point_features: torch.Tensor) -> torch.Tensor:
        model_inputs = self.point_model_inputs(points, point_features)
        hidden = self.backbone(model_inputs, xyz=point_geometry(model_inputs), t=None)
        return self.head(hidden)

    def predict_diffusion_target(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(torch.cat([x_t, cond], dim=-1), xyz=point_geometry(cond), t=t)
        return self.head(hidden)

    def predict_point_labels(
        self,
        points: torch.Tensor,
        point_features: torch.Tensor,
        valid_geometry: torch.Tensor,
    ) -> torch.Tensor:
        return self.behavior_impl.predict_point_labels(
            self,
            points,
            point_features,
            valid_geometry,
        )

    def _compute_stage_output(
        self,
        batch: dict,
        *,
        stage: str,
        evaluation: bool,
    ) -> dict:
        return self.behavior_impl.compute_point_stage_output(
            self,
            batch,
            stage=stage,
            evaluation=evaluation,
        )

    def configure_optimizers(self):
        return self.behavior_impl.configure_optimizers(self)


class RangeImageTaskModel(RangeImageSegmentationModel):
    def __init__(
        self,
        num_classes: int = 23,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        backbone: str = "unet",
        head: str = "segmentation",
        behavior: str = "supervised",
        base_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.0,
        diffusion_steps: int = 1000,
        use_balanced_class_weights: bool = True,
        geometry_only: bool = False,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            use_balanced_class_weights=use_balanced_class_weights,
        )
        input_channels = range_input_channels(geometry_only=bool(geometry_only))
        self.save_hyperparameters()
        self.behavior_impl = build_range_behavior(behavior, diffusion_steps=int(diffusion_steps))
        self.backbone = build_range_backbone(
            str(backbone),
            input_channels=int(input_channels),
            num_classes=int(num_classes),
            base_channels=int(base_channels),
            depth=int(depth),
            dropout=float(dropout),
        )
        self.head = _build_range_head(str(head), input_channels=int(self.backbone.output_channels), output_channels=int(num_classes))

    def prepare_runtime(self) -> None:
        self.behavior_impl.prepare_runtime(self)

    def prepare_range_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = select_range_model_inputs(batch["range_images"], geometry_only=bool(self.hparams.geometry_only))
        labels = batch["semantic"].long()
        valid = batch["valid_label"].bool()
        if x.ndim != 5:
            raise ValueError(f"Expected selected range input shape [B,R,H,W,C], got {tuple(x.shape)}")
        bsz, returns, height, width, channels = x.shape
        x = x.permute(0, 1, 4, 2, 3).reshape(bsz * returns, channels, height, width)
        labels = labels.reshape(bsz * returns, height, width)
        valid = valid.reshape(bsz * returns, height, width)
        return x, labels, valid

    def predict_range_logits(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, labels, valid = self.prepare_range_batch(batch)
        hidden = self.backbone(x, cond=None, t=None)
        logits = self.head(hidden)
        return logits, labels, valid & (labels > 0)

    def predict_diffusion_target(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(x_t, cond=cond, t=t)
        return self.head(hidden)

    def predict_range_labels(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        return self.behavior_impl.predict_range_labels(self, batch)

    def _compute_stage_output(
        self,
        batch: dict,
        *,
        stage: str,
        evaluation: bool,
    ) -> dict:
        return self.behavior_impl.compute_range_stage_output(
            self,
            batch,
            stage=stage,
            evaluation=evaluation,
        )

    def configure_optimizers(self):
        return self.behavior_impl.configure_optimizers(self)
