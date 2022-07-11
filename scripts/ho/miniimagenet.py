import datetime
from pathlib import Path

import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import aim
from aim.pytorch_lightning import AimLogger

from mixturevqvae.datasets import MiniImagenetDataModule
from mixturevqvae.prior.ho.diffusion_noise import DenoiseDiffusion
from mixturevqvae.prior.ho.layers.unet import UNet
from mixturevqvae.vqvae import MiniImagenetVQVAE
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


class LitHoMiniimagenet(pl.LightningModule):
    beta_start: float = 0.0001
    beta_end: float = 0.02
    path_vqvae_weights: Path = Path("vqvae_miniimagenet.pt")

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
        self.vqvae = MiniImagenetVQVAE(
            num_codebook=num_codebook, dim_codebook=dim_codebook
        )
        try:
            self.vqvae.load_state_dict(torch.load(self.path_vqvae_weights))
            print(f"Loaded vqvae weights at {self.path_vqvae_weights}.")
        except FileNotFoundError:
            pass
        # Different attention usage in UNet as in legacy model
        channel_multipliers = [1, 2, 4]
        is_attention = [False, False, True]
        self.eps_model = UNet(
            image_channels=dim_codebook,
            n_channels=n_channels,
            ch_mults=channel_multipliers,
            is_attn=is_attention,
        )
        self.diffusion_model = DenoiseDiffusion(
            eps_model=self.eps_model,
            n_steps=num_steps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.eps_model.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        encodings = self.vqvae.encode(batch).detach()
        loss = self.diffusion_model.loss(encodings)
        self.log("train_loss", loss)
        # Sample images on the first batch
        if batch_idx == 0 and self.current_epoch % 10 == 0:
            nll = self.compute_patched_nll(encodings, subset="train")
            self.log("train_nll", nll)
        return loss

    def validation_step(self, batch, batch_idx):
        encodings = self.vqvae.encode(batch)
        loss = self.diffusion_model.loss(encodings)
        self.log("val_loss", loss)
        # Sample images on the first batch
        if batch_idx == 0 and self.current_epoch % 10 == 0:
            # Sample from the prior
            x_T = torch.randn_like(encodings)
            self._log_images(self.sample_images(x_T), name="samples")
            # Compute NLL
            nll = self.compute_patched_nll(encodings)
            self.log("val_nll", nll)

    @torch.no_grad()
    def compute_patched_nll(self, encodings, subset="val"):
        # Generate mask
        patch = middle_patch_coordinates(self.vqvae.featuremap_size[0], d=9)
        i_0, j_0, i_1, j_1 = patch
        mask = torch.zeros(encodings.shape, device=encodings.device)
        mask[..., i_0:i_1, j_0:j_1] = 1
        # Compute observed information for all t
        time_steps = (
            torch.arange(
                self.hparams.num_steps, device=encodings.device, dtype=torch.long
            )
            .unsqueeze(-1)
            .tile((1, encodings.size(0)))
        )
        observations = [encodings]
        for t in time_steps:
            observations.append(
                self.diffusion_model.q_xtm1_given_xt(observations[-1], t)
            )
        # Sample loop from the prior model
        x = (1 - mask) * observations[-1] + mask * torch.randn_like(observations[-1])
        for t, x_observed in reversed(list(zip(time_steps, observations))):
            x = self.diffusion_model.p_sample(xt=x, t=t)
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

    def q_sample(self, x0: torch.Tensor, t: int) -> torch.Tensor:
        """Sample from the posterior at time `t`.

        Sample from $q_t(x_t | x_0)$.

        Parameters
        ----------
        x0: `x_0` tensor.
        t: time step index.

        Returns
        -------
        A sample from the posterior.
        """
        t = torch.full(
            size=(x0.size(0),),
            fill_value=t,
            device=x0.device,
            dtype=torch.long,
        )
        return self.diffusion_model.q_sample(x0=x0, t=t)

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
        return self.vqvae.decode(zq.permute(0, 3, 1, 2)).detach()

    @torch.no_grad()
    def sample_images(self, x: torch.Tensor) -> torch.Tensor:
        """Sample images.

        Iteratively sample from $p_\theta(x_{t-1}|x_t)$ and quantize and decode.

        Parameters
        ----------
        x: `x_T`, initialization of the sampling loop. If `None`, `x_T` is sampled from
        a normal distribution. Default is `None`.
        """
        time_steps = (
            torch.arange(self.hparams.num_steps, device=x.device, dtype=torch.long)
            .unsqueeze(-1)
            .tile((1, x.size(0)))
        )
        time_steps = self.hparams.num_steps - time_steps - 1
        for t in time_steps:
            x = self.diffusion_model.p_sample(xt=x, t=t)
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
    exp_name = "ho-miniimagenet-train"
    dataset_path = "./datasets/miniimagenet/"
    num_steps: int = 1000
    n_channels: int = 128
    num_codebook: int = 64
    dim_codebook: int = 32
    litmodule = LitHoMiniimagenet

    def __init__(self, args):
        self.datamodule = MiniImagenetDataModule(
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
