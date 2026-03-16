from typing import Optional

import torch
import torch.nn.functional as F


class DDPM:
    def __init__(
        self,
        T: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = "cpu",
    ) -> None:
        self.T = int(T)
        self.device = device

        betas = torch.linspace(beta_start, beta_end, self.T, dtype=torch.float64)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = F.pad(alpha_bars[:-1], (1, 0), value=1.0)

        self.register("betas", betas.float())
        self.register("alphas", alphas.float())
        self.register("alpha_bars", alpha_bars.float())
        self.register("alpha_bars_prev", alpha_bars_prev.float())
        self.register("sqrt_alpha_bars", alpha_bars.sqrt().float())
        self.register("sqrt_one_minus_ab", (1 - alpha_bars).sqrt().float())
        self.register("sqrt_recip_alphas", (1.0 / alphas).sqrt().float())

        beta_tilde = betas * (1 - alpha_bars_prev) / (1 - alpha_bars)
        self.register("beta_tilde", beta_tilde.float())
        self.register("sqrt_beta_tilde", beta_tilde.sqrt().float())

    def register(self, name: str, tensor: torch.Tensor) -> None:
        setattr(self, name, tensor)

    def to(self, device: torch.device | str) -> "DDPM":
        self.device = str(device)
        for name in [
            "betas",
            "alphas",
            "alpha_bars",
            "alpha_bars_prev",
            "sqrt_alpha_bars",
            "sqrt_one_minus_ab",
            "sqrt_recip_alphas",
            "beta_tilde",
            "sqrt_beta_tilde",
        ]:
            setattr(self, name, getattr(self, name).to(device))
        return self

    def _extract(self, schedule: torch.Tensor, t: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        t_idx = (t - 1).clamp(0, len(schedule) - 1)
        vals = schedule.gather(0, t_idx)
        return vals.view(t.shape[0], *([1] * (len(shape) - 1)))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        if eps is None:
            eps = torch.randn_like(x0)
        sqrt_ab = self._extract(self.sqrt_alpha_bars, t, x0.shape)
        sqrt_1mab = self._extract(self.sqrt_one_minus_ab, t, x0.shape)
        return sqrt_ab * x0 + sqrt_1mab * eps, eps

    def loss(
        self,
        eps_pred: torch.Tensor,
        eps_true: torch.Tensor,
        valid: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
        class_dim: int = 1,
    ) -> torch.Tensor:
        if labels is not None:
            mask = (valid & (labels > 0)).float()
        else:
            mask = valid.float()
        mask = mask.unsqueeze(class_dim)

        diff = (eps_pred - eps_true) ** 2
        diff = diff * mask

        if class_weights is not None and labels is not None:
            pixel_weights = class_weights[labels.clamp(0, class_weights.shape[0] - 1)].float()
            pixel_weights = pixel_weights.unsqueeze(class_dim)
            diff = diff * pixel_weights
            denom = (pixel_weights * mask).sum().clamp(min=1.0)
        else:
            denom = mask.sum().clamp(min=1.0)
        return diff.sum() / denom

    @torch.no_grad()
    def p_sample(self, model, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        eps_pred = model(x_t, t, cond)
        recip_sqrt_alpha = self._extract(self.sqrt_recip_alphas, t, x_t.shape)
        beta_t = self._extract(self.betas, t, x_t.shape)
        sqrt_1mab = self._extract(self.sqrt_one_minus_ab, t, x_t.shape)
        mu = recip_sqrt_alpha * (x_t - (beta_t / sqrt_1mab) * eps_pred)

        sqrt_beta_tilde = self._extract(self.sqrt_beta_tilde, t, x_t.shape)
        z = torch.randn_like(x_t)
        nonzero = (t > 1).float().view(t.shape[0], *([1] * (len(x_t.shape) - 1)))
        return mu + sqrt_beta_tilde * z * nonzero

    @torch.no_grad()
    def sample(self, model, cond: torch.Tensor, sample_shape: tuple[int, ...]) -> torch.Tensor:
        x_t = torch.randn(sample_shape, device=cond.device)
        for t_int in range(self.T, 0, -1):
            t = torch.full((sample_shape[0],), int(t_int), dtype=torch.long, device=cond.device)
            x_t = self.p_sample(model, x_t, t, cond)
        return x_t


def labels_to_soft_range(labels: torch.Tensor, num_classes: int, valid: torch.Tensor) -> torch.Tensor:
    one_hot = F.one_hot(labels.clamp(0, num_classes - 1), num_classes)
    x0 = one_hot.permute(0, 3, 1, 2).float() * 2.0 - 1.0
    return x0 * valid.unsqueeze(1).float()


def labels_to_soft_points(labels: torch.Tensor, num_classes: int, valid: torch.Tensor) -> torch.Tensor:
    one_hot = F.one_hot(labels.clamp(0, num_classes - 1), num_classes).float()
    return (one_hot * 2.0 - 1.0) * valid.unsqueeze(-1).float()
