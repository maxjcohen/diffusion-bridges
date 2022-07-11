import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import aim
import numpy as np
import matplotlib.pyplot as plt

from ddpm.diffusion_bridge import DiffusionBridge

from mixturevqvae.prior.bridge.toy.toy_diffusion import Net, SphericalProjection


class ToyModel(DiffusionBridge):
    theta = 2
    eta2 = 0.01
    diffusion_target = 0

    def __init__(
        self,
        denoising_model: callable,
        delta_schedule: torch.Tensor,
        num_steps: int,
        K,
        dim,
    ):
        super().__init__(
            denoising_model=denoising_model,
            delta_schedule=delta_schedule,
            num_steps=num_steps,
            theta=self.theta,
            eta2=self.eta2,
            diffusion_target=self.diffusion_target,
        )
        self.sp = SphericalProjection(K=K, m=dim)


class LitToyModel(pl.LightningModule):
    def __init__(
        self, denoising_model, delta_schedule, num_steps=20, K=8, dim=2, lr=1e-3
    ):
        super().__init__()
        self.save_hyperparameters("num_steps", "K", "dim", "lr")
        self.toy_model = ToyModel(
            denoising_model, delta_schedule, num_steps, K=K, dim=dim
        )

    def training_step(self, batch, batch_idx):
        loss = self.toy_model.compute_loss(batch)
        self.log("train_loss", loss)
        if batch_idx == 0:
            self.compute_elbo(batch)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    @torch.no_grad()
    def compute_elbo(self, encodings):
        def compute_Lt(encodings, t):
            batch_size = encodings.shape[0]
            # Reparametrize encodings_t
            noise = torch.randn_like(encodings)
            encodings_t = self.toy_model.posterior_sample(encodings, t, epsilon=noise)
            # Compute eps_theta
            eps_theta = self.toy_model.denoising_model(encodings_t, t)
            # Compute difference between noises
            diff = (
                F.mse_loss(noise, eps_theta, reduction="none")
                .flatten(start_dim=1)
                .sum(-1)
            )
            return diff

        time_steps = (
            torch.arange(
                2, self.hparams.num_steps, dtype=torch.long, device=encodings.device
            )
            .unsqueeze(-1)
            .tile(encodings.shape[0])
        )
        elbo_prior = torch.stack([compute_Lt(encodings, t) for t in time_steps])
        # Plot elbo_prior at time steps
        fig = plt.figure()
        plt.plot(time_steps[:, 0].cpu().numpy(), elbo_prior.mean(-1).cpu().numpy())
        plt.close(fig)
        self.logger.experiment.track(
            aim.Figure(fig), step=self.current_epoch, name="elbo prior"
        )
