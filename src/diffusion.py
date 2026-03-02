"""
diffusion.py
============
DDPM noise schedule and sampling, taken directly from the lecture slides
(Ho et al. 2020) with minor adaptations for our conditional segmentation task.

Key differences from vanilla DDPM:
  - x_0 is a soft label map (23, H, W), not an RGB image
  - The U-Net is conditioned on the LiDAR range image (4, H, W)
  - Loss is masked to only backprop on valid pixels (range > 0)
  - Optional per-class inverse-frequency weighting to handle class imbalance
  - No latent encoding step — we work directly in range image space
"""

import torch
import torch.nn.functional as F


class DDPM:
    """
    DDPM noise schedule and forward/reverse process.

    From slides:
        beta_t : linearly from beta_start=1e-4 to beta_end=0.02
        alpha_t       = 1 - beta_t
        alpha_bar_t   = prod_{i=1}^{t} alpha_i   (cumulative product)

        Forward (noising):
            x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
            where eps ~ N(0, I)

        Reverse (denoising, one step):
            mu_theta = (1/sqrt(alpha_t)) * (x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * eps_theta)
            x_{t-1} = mu_theta + sqrt(beta_tilde_t) * z
            where z ~ N(0,I) if t > 1 else z = 0
            beta_tilde_t = (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) * beta_t

        Loss (simple MSE from slides, no weighting term):
            L = E[|| eps - eps_theta(x_t, t) ||^2]
            masked to valid pixels only
            optionally weighted by inverse class frequency
    """

    def __init__(
        self,
        T: int   = 1000,
        beta_start: float = 1e-4,
        beta_end:   float = 0.02,
        device: str = "cpu",
    ):
        self.T      = T
        self.device = device

        # Linear beta schedule (slide: increasing from 1e-4 to 0.02)
        betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float64)

        alphas          = 1.0 - betas
        alpha_bars      = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = F.pad(alpha_bars[:-1], (1, 0), value=1.0)

        # Precompute all quantities used in training and sampling
        self.register("betas",           betas.float())
        self.register("alphas",          alphas.float())
        self.register("alpha_bars",      alpha_bars.float())
        self.register("alpha_bars_prev", alpha_bars_prev.float())

        self.register("sqrt_alpha_bars",        alpha_bars.sqrt().float())
        self.register("sqrt_one_minus_ab",      (1 - alpha_bars).sqrt().float())
        self.register("sqrt_recip_alphas",      (1.0 / alphas).sqrt().float())

        # beta_tilde_t = (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) * beta_t
        beta_tilde = betas * (1 - alpha_bars_prev) / (1 - alpha_bars)
        self.register("beta_tilde",      beta_tilde.float())
        self.register("sqrt_beta_tilde", beta_tilde.sqrt().float())

    def register(self, name: str, tensor: torch.Tensor):
        """Store a schedule tensor as an attribute (moved to device on demand)."""
        setattr(self, name, tensor)

    def to(self, device):
        self.device = device
        for attr in ["betas", "alphas", "alpha_bars", "alpha_bars_prev",
                     "sqrt_alpha_bars", "sqrt_one_minus_ab", "sqrt_recip_alphas",
                     "beta_tilde", "sqrt_beta_tilde"]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    def _extract(self, schedule: torch.Tensor, t: torch.Tensor, shape) -> torch.Tensor:
        """
        Gather schedule values at timesteps t and reshape to broadcast over (B, C, H, W).
        t : (B,) long, 1-indexed (1 to T) — clamped to [0, T-1] for gather.
        """
        t_idx = (t - 1).clamp(0, len(schedule) - 1)  # convert 1-indexed → 0-indexed
        vals  = schedule.gather(0, t_idx)
        return vals.view(t.shape[0], *([1] * (len(shape) - 1)))

    # ── Forward process ────────────────────────────────────────────────────────
    def q_sample(
        self,
        x0: torch.Tensor,   # (B, C, H, W)  clean label map (soft / one-hot)
        t:  torch.Tensor,   # (B,)           timestep indices
        eps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample x_t given x_0 using the closed-form forward process:
            x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps

        Returns (x_t, eps) — eps is returned so we can compute the loss.
        """
        if eps is None:
            eps = torch.randn_like(x0)

        sqrt_ab   = self._extract(self.sqrt_alpha_bars,   t, x0.shape)
        sqrt_1mab = self._extract(self.sqrt_one_minus_ab, t, x0.shape)

        x_t = sqrt_ab * x0 + sqrt_1mab * eps
        return x_t, eps

    # ── Loss ───────────────────────────────────────────────────────────────────
    def loss(
        self,
        eps_pred:     torch.Tensor,              # (B, C, H, W)  U-Net output
        eps_true:     torch.Tensor,              # (B, C, H, W)  sampled noise
        valid:        torch.Tensor,              # (B, H, W)     bool mask
        labels:       torch.Tensor | None = None,  # (B, H, W)  int64 true class
        class_weights: torch.Tensor | None = None, # (C,)       per-class weights
    ) -> torch.Tensor:
        """
        MSE loss on epsilon, masked to valid pixels only.
        Optionally weighted by per-class inverse frequency weights.

        From slides:
            L_simple = E[|| eps - eps_theta(x_t, t) ||^2]

        With class weighting:
            L = E[w(c) * || eps - eps_theta(x_t, t) ||^2]
            where w(c) = inverse frequency weight for the true class c at each pixel.
            Rare classes (motorcyclist, bicyclist) get upweighted so the model
            is penalized more for getting them wrong.
        """
        # Mask: valid range pixels AND not undefined (label 0)
        if labels is not None:
            defined_mask = (labels != 0)
            mask = (valid & defined_mask).unsqueeze(1).float()  # (B, 1, H, W)
        else:
            mask = valid.unsqueeze(1).float()                   # (B, 1, H, W)

        diff = (eps_pred - eps_true) ** 2          # (B, C, H, W)
        diff = diff * mask                          # zero out invalid/undefined pixels

        if class_weights is not None and labels is not None:
            # pixel_weights: (B, H, W) — one scalar weight per pixel from its true class
            # class_weights[0] = 0.0 so undefined pixels contribute nothing
            pixel_weights = class_weights[labels.clamp(0, class_weights.shape[0] - 1)]  # (B, H, W)
            pixel_weights = pixel_weights.unsqueeze(1)             # (B, 1, H, W)
            diff = diff * pixel_weights
            n_valid = (pixel_weights * mask).sum().clamp(min=1.0)
        else:
            n_valid = mask.sum().clamp(min=1.0)

        return diff.sum() / n_valid

    # ── Reverse process (single step) ─────────────────────────────────────────
    @torch.no_grad()
    def p_sample(
        self,
        model,
        x_t:    torch.Tensor,   # (B, C, H, W)  noisy label map at step t
        t:      torch.Tensor,   # (B,)           current timestep
        cond:   torch.Tensor,   # (B, 4, H, W)   LiDAR range image condition
    ) -> torch.Tensor:
        """
        One reverse denoising step:
            mu_theta = (1/sqrt(alpha_t)) * (x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * eps_theta)
            x_{t-1}  = mu_theta + sqrt(beta_tilde_t) * z    (z=0 at t=1)
        """
        eps_pred = model(x_t, t, cond)

        recip_sqrt_alpha = self._extract(self.sqrt_recip_alphas, t, x_t.shape)
        beta_t           = self._extract(self.betas,             t, x_t.shape)
        sqrt_1mab        = self._extract(self.sqrt_one_minus_ab, t, x_t.shape)

        mu = recip_sqrt_alpha * (x_t - (beta_t / sqrt_1mab) * eps_pred)

        sqrt_beta_tilde = self._extract(self.sqrt_beta_tilde, t, x_t.shape)
        z = torch.randn_like(x_t)
        nonzero = (t > 1).float().view(t.shape[0], *([1] * (len(x_t.shape) - 1)))
        z = z * nonzero

        return mu + sqrt_beta_tilde * z

    # ── Full reverse sampling ──────────────────────────────────────────────────
    @torch.no_grad()
    def sample(
        self,
        model,
        cond:        torch.Tensor,   # (B, 4, H, W) LiDAR condition
        num_classes: int = 23,
    ) -> torch.Tensor:
        """
        Full reverse diffusion: start from x_T ~ N(0,I), iteratively denoise.
        Returns x_0: (B, num_classes, H, W) — the predicted soft label map.
        Take argmax over dim=1 for the final segmentation prediction.
        """
        B, _, H, W = cond.shape
        device      = cond.device

        x_t = torch.randn(B, num_classes, H, W, device=device)

        for t_int in reversed(range(1, self.T + 1)):
            t = torch.full((B,), t_int, device=device, dtype=torch.long)
            x_t = self.p_sample(model, x_t, t, cond)

        return x_t