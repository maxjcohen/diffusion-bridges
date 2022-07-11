import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from aim.pytorch_lightning import AimLogger
import matplotlib.pyplot as plt
import aim

from mixturevqvae.datasets import CifarDataModule
from mixturevqvae.prior.ho import HoDiffusion as DenoiseDiffusion
from mixturevqvae.prior.ho.layers.unet import UNet
from mixturevqvae.vqvae import CifarVQVAE
from mixturevqvae.utils.patch import middle_patch_coordinates, mask_patch
from ..utils import parser


parser.set_defaults(lr=2e-5, batch_size=32, epochs=1000)


def image_compare_reconstructions(originals, reconstructions):
    image = torch.cat(
        [
            torch.cat([original, reconstruction], dim=1)
            for original, reconstruction in zip(originals, reconstructions)
        ],
        dim=2,
    )
    image = image.clip(min=0, max=1)
    return image


class LitHo(pl.LightningModule):
    beta_start: float = 0.0001
    beta_end: float = 0.02
    path_vqvae_weights: Path = Path("vqvae_cifar.pt")

    def __init__(
        self,
        num_steps: int,
        n_channels: int,
        num_codebook: int,
        dim_codebook: int,
        lr: float,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.vqvae = CifarVQVAE(num_codebook=num_codebook, dim_codebook=dim_codebook)
        try:
            self.vqvae.load_state_dict(torch.load(self.path_vqvae_weights))
            print(f"Loaded vqvae weights at {self.path_vqvae_weights}.")
        except FileNotFoundError:
            pass
        channel_multipliers = [1, 2, 4]
        is_attention = [False, False, True]
        self.eps_model = UNet(
            image_channels=dim_codebook,
            n_channels=n_channels,
            ch_mults=channel_multipliers,
            is_attn=is_attention,
        )
        beta = torch.linspace(self.beta_start, self.beta_end, num_steps)
        delta = - torch.log(1 - beta) / 2
        self.diffusion_model = DenoiseDiffusion(
            denoising_model=self.eps_model,
            delta_schedule=delta,
            num_steps=num_steps,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.eps_model.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        encodings = self.vqvae.encode(batch).detach()
        loss = self.diffusion_model.compute_loss(encodings)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        encodings = self.vqvae.encode(batch)
        loss = self.diffusion_model.compute_loss(encodings)
        self.log("val_loss", loss)
        # Sample images on the first batch
        if batch_idx == 0:
            elbo = self.compute_elbo(batch)
            self.log("val_elbo", elbo)
        if batch_idx == 0 and self.current_epoch % 10 == 0:
            # Sample from the prior
            x_T = torch.randn_like(encodings)
            self._log_images(self.sample_images(x_T), name="samples")
            # Compute NLL
            nll = self.compute_patched_nll(encodings)
            self.log("val_nll_inpainting", nll)

    @torch.no_grad()
    def compute_elbo(self, images):
        def q_phi_sample(encodings, temperature=0.01):
            dist = self.vqvae.codebook.compute_distances(encodings.permute(0, 2, 3, 1))
            probas = F.softmax(-dist / temperature, dim=-1)
            logits = torch.distributions.Multinomial(probs=probas).sample()
            quantized = self.vqvae.codebook(logits.argmax(-1))
            return quantized, logits

        def compute_elbo_reconstruction(images, quantized):
            reconstructions = self.vqvae.decode(quantized.permute(0, 3, 1, 2))
            diff = (images - reconstructions).square().flatten(start_dim=1).sum(-1)
            return -0.5 * (np.log(2 * np.pi) + diff)

        def compute_Lt(encodings, t):
            batch_size = encodings.shape[0]
            # Reparametrize encodings_t
            noise = torch.randn_like(encodings)
            encodings_t = self.diffusion_model.posterior_sample(encodings, t, epsilon=noise)
            # Compute eps_theta
            eps_theta = self.eps_model(encodings_t, t)
            # Compute difference between noises
            diff = (
                F.mse_loss(noise, eps_theta, reduction="none")
                .flatten(start_dim=1)
                .sum(-1)
            )
            coeff = -self.diffusion_model.beta[t-1] / (
                2
                * self.diffusion_model.alpha[t-1]
                * (1 - self.diffusion_model.alpha_bar[t-1])
            )
            Lt = coeff * diff
            return Lt

        # Sample z_q under q_phi
        encodings = self.vqvae.encode(images)
        quantized, logits = q_phi_sample(encodings)
        # Compute reconstruction term
        elbo_reconstruction = compute_elbo_reconstruction(images, quantized)
        # Compute prior term
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
        # Sum on all time steps
        elbo_prior = elbo_prior.sum(0)
        elbo = elbo_reconstruction + elbo_prior
        return elbo.mean()

    @torch.no_grad()
    def compute_patched_nll(self, encodings, subset="val"):
        # Generate mask
        patch = middle_patch_coordinates(self.vqvae.featuremap_size[0], d=7)
        i_0, j_0, i_1, j_1 = patch
        mask = torch.zeros(encodings.shape, device=encodings.device)
        mask[..., i_0:i_1, j_0:j_1] = 1
        # Compute observed information for all t
        time_steps = (
            torch.arange(
                1, self.hparams.num_steps+1, device=encodings.device, dtype=torch.long
            )
            .unsqueeze(-1)
            .tile((1, encodings.size(0)))
        )
        observations = [encodings]
        for t in time_steps:
            observations.append(
                self.diffusion_model.posterior_onestep_sample(x=observations[-1], t=t)
            )
        # Sample loop from the prior model
        x = (1 - mask) * observations[-1] + mask * torch.randn_like(observations[-1])
        for t, x_observed in reversed(list(zip(time_steps-1, observations))):
            x = self.diffusion_model.prior_onestep_sample(x=x, t=t)
            x = (1 - mask) * x_observed + mask * x
        samples = self.ze_to_reconstruction(x)
        self.logger.experiment.track(
            aim.Image(
                image_compare_reconstructions(
                    self.ze_to_reconstruction(mask_patch(encodings, patch)), samples
                )
            ),
            name="inpainting",
            epoch=self.current_epoch,
            context={"subset": subset},
        )
        # Compute NLL
        _, indices, _ = self.vqvae.codebook.quantize(encodings.permute(0, 2, 3, 1))
        indices = indices.argmax(-1)
        dist = self.vqvae.codebook.compute_distances(x.permute(0, 2, 3, 1))
        dist = dist.permute(0, 3, 1, 2)
        nll = torch.nn.functional.cross_entropy(-dist, indices, reduction="none")
        nll = nll[..., i_0:i_1, j_0:j_1].mean() / np.log(2)
        return nll

    @torch.no_grad()
    def ze_to_reconstruction(self, ze):
        """Quantize ze and returns decoding.

        Parameters
        ----------
        ze: latent tensor with shape `(batch_size, latent_dim, width, height)`.

        Returns
        -------
        Decoded tensor with shape `(batch_size, channels, width, height)`.
        """
        zq, *_ = self.vqvae.codebook.quantize(ze.permute(0, 2, 3, 1))
        return self.vqvae.decode(zq.permute(0, 3, 1, 2))

    @torch.no_grad()
    def sample_images(self, x: torch.Tensor) -> torch.Tensor:
        """Sample images.

        Iteratively sample from $p_\theta(x_{t-1}|x_t)$ and quantize and decode.

        Parameters
        ----------
        x: `x_T`, initialization of the sampling loop. If `None`, `x_T` is sampled from
        a normal distribution. Default is `None`.
        """
        x = self.diffusion_model.prior_sample(shape=x.shape)
        samples = self.ze_to_reconstruction(x)
        return samples

    def _log_images(self, images, name: str, context: str = "val"):
        self.logger.experiment.track(
            [aim.Image(image.cpu().clip(0, 1)) for image in images],
            name=name,
            context={"subset": context},
            epoch=self.current_epoch,
        )


class Experiment:
    exp_name = "ho-cifar-train"
    dataset_path = "./datasets/cifar"
    num_steps: int = 1000
    n_channels: int = 128
    num_codebook: int = 64
    dim_codebook: int = 32
    litmodule = LitHo

    def __init__(self, args):
        self.datamodule = CifarDataModule(
            dataset_path=self.dataset_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        # Load LitModule
        if args.load_checkpoint is not None:
            self.litmodule = self.litmodule.load_from_checkpoint(args.load_checkpoint)
            print("Loaded checkpoint.")
        else:
            self.litmodule = self.litmodule(
                num_steps=self.num_steps,
                n_channels=self.n_channels,
                num_codebook=self.num_codebook,
                dim_codebook=self.dim_codebook,
                lr=args.lr,
            )

        # Load trainer
        self.logger = AimLogger(
            experiment=self.exp_name,
            system_tracking_interval=None,
            log_system_params=False,
        )
        self.logger.experiment["hparams"] = vars(args)
        checkpoint_callback = ModelCheckpoint(
            dirpath=Path("checkpoints") / self.exp_name,
            filename=f"{datetime.datetime.now().strftime('%Y_%m_%d__%H%M%S')}",
            monitor="val_loss",
            save_last=True,
        )
        self.trainer = pl.Trainer(
            max_epochs=args.epochs,
            gpus=args.gpus,
            logger=self.logger,
            callbacks=[checkpoint_callback],
            limit_val_batches=args.limit_val_batches,
        )


if __name__ == "__main__":
    args = parser.parse_args()
    exp = Experiment(args)
    exp.trainer.fit(exp.litmodule, datamodule=exp.datamodule)
