import torch

from ddpm.diffusion_bridge import DiffusionBridge


class HoDiffusion(DiffusionBridge):
    """DDPM as defined in Ho https://arxiv.org/pdf/2006.11239.pdf."""

    theta = 1
    eta2 = 2
    diffusion_target = 0

    def __init__(
        self,
        denoising_model: callable,
        delta_schedule: torch.Tensor,
        num_steps: int,
    ):
        super().__init__(
            denoising_model=denoising_model,
            delta_schedule=delta_schedule,
            num_steps=num_steps,
            theta=self.theta,
            eta2=self.eta2,
            diffusion_target=self.diffusion_target,
        )
