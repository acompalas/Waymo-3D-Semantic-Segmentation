"""Core DDPM utilities for point-cloud semantic segmentation diffusion."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


class DDPM:
    """DDPM schedule + forward/reverse process for tensors shaped (B, N, C)."""

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

    def to(self, device: torch.device) -> "DDPM":
        self.device = str(device)
        names = [
            "betas",
            "alphas",
            "alpha_bars",
            "alpha_bars_prev",
            "sqrt_alpha_bars",
            "sqrt_one_minus_ab",
            "sqrt_recip_alphas",
            "beta_tilde",
            "sqrt_beta_tilde",
        ]
        for name in names:
            setattr(self, name, getattr(self, name).to(device))
        return self

    def _extract(self, schedule: torch.Tensor, t: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
        t_idx = (t - 1).clamp(0, len(schedule) - 1)
        vals = schedule.gather(0, t_idx)
        return vals.view(t.shape[0], *([1] * (len(shape) - 1)))

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        eps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if eps is None:
            eps = torch.randn_like(x0)

        sqrt_ab = self._extract(self.sqrt_alpha_bars, t, x0.shape)
        sqrt_1mab = self._extract(self.sqrt_one_minus_ab, t, x0.shape)
        x_t = sqrt_ab * x0 + sqrt_1mab * eps
        return x_t, eps

    def loss(
        self,
        eps_pred: torch.Tensor,
        eps_true: torch.Tensor,
        valid: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # valid: (B, N) bool
        if labels is not None:
            defined = labels != 0
            mask = (valid & defined).unsqueeze(-1).float()
        else:
            mask = valid.unsqueeze(-1).float()

        diff = (eps_pred - eps_true) ** 2
        diff = diff * mask

        if class_weights is not None and labels is not None:
            pix_w = class_weights[labels.clamp(0, class_weights.shape[0] - 1)].unsqueeze(-1)
            diff = diff * pix_w
            denom = (pix_w * mask).sum().clamp(min=1.0)
        else:
            denom = mask.sum().clamp(min=1.0)

        return diff.sum() / denom

    @torch.no_grad()
    def p_sample(self, model, x_t: torch.Tensor, t: torch.Tensor, cond_points: torch.Tensor) -> torch.Tensor:
        eps_pred = model(x_t, t, cond_points)

        recip_sqrt_alpha = self._extract(self.sqrt_recip_alphas, t, x_t.shape)
        beta_t = self._extract(self.betas, t, x_t.shape)
        sqrt_1mab = self._extract(self.sqrt_one_minus_ab, t, x_t.shape)

        mu = recip_sqrt_alpha * (x_t - (beta_t / sqrt_1mab) * eps_pred)

        sqrt_beta_tilde = self._extract(self.sqrt_beta_tilde, t, x_t.shape)
        z = torch.randn_like(x_t)
        nonzero = (t > 1).float().view(t.shape[0], *([1] * (len(x_t.shape) - 1)))
        z = z * nonzero

        return mu + sqrt_beta_tilde * z

    @torch.no_grad()
    def sample(self, model, cond_points: torch.Tensor, num_classes: int = 23) -> torch.Tensor:
        bsz, npts, _ = cond_points.shape
        device = cond_points.device

        x_t = torch.randn(bsz, npts, num_classes, device=device)

        for t_int in reversed(range(1, self.T + 1)):
            t = torch.full((bsz,), t_int, dtype=torch.long, device=device)
            x_t = self.p_sample(model, x_t, t, cond_points)

        return x_t


def labels_to_soft(labels: torch.Tensor, num_classes: int, valid: torch.Tensor) -> torch.Tensor:
    """Integer labels (B,N) to soft one-hot in [-1,1], invalid points zeroed."""
    one_hot = F.one_hot(labels.clamp(0, num_classes - 1), num_classes).float()
    x0 = one_hot * 2.0 - 1.0
    x0 = x0 * valid.unsqueeze(-1).float()
    return x0
