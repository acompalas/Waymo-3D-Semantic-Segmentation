from .diffusion import DDPM


class SupervisedBehavior:
    time_dim: int | None = None

    @staticmethod
    def point_backbone_input_dim(*, point_input_dim: int, num_classes: int) -> int:
        _ = num_classes
        return int(point_input_dim)

    def prepare_runtime(self, model) -> None:
        _ = model

    def compute_stage_output(self, model, batch: dict, *, evaluation: bool) -> dict:
        _ = evaluation
        return model.compute_supervised_stage_output(batch)


class DiffusionBehavior:
    time_dim: int | None = 128

    def __init__(self, diffusion_steps: int) -> None:
        self.ddpm = DDPM(T=int(diffusion_steps))

    @staticmethod
    def point_backbone_input_dim(*, point_input_dim: int, num_classes: int) -> int:
        return int(point_input_dim) + int(num_classes)

    def prepare_runtime(self, model) -> None:
        self.ddpm.to(model.device)

    def compute_stage_output(self, model, batch: dict, *, evaluation: bool) -> dict:
        if evaluation:
            return model.compute_diffusion_evaluation_stage_output(batch, self.ddpm)
        return model.compute_diffusion_training_stage_output(batch, self.ddpm)


def build_behavior(behavior: str, *, diffusion_steps: int):
    key = str(behavior).lower()
    if key == "supervised":
        return SupervisedBehavior()
    if key == "diffusion":
        return DiffusionBehavior(diffusion_steps=int(diffusion_steps))
    raise ValueError(f"Unsupported behavior '{behavior}'.")
