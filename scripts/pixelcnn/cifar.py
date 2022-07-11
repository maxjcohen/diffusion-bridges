import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from aim import Image
from aim.pytorch_lightning import AimLogger

from mixturevqvae.prior.pixelcnn import GatedPixelCNN
from mixturevqvae.vqvae import CifarVQVAE, MiniImagenetVQVAE
from mixturevqvae.datasets import CifarDataModule
from mixturevqvae.utils.patch import middle_patch_coordinates, mask_patch
from ..utils import parser

parser.set_defaults(lr=1e-4)
parser.add_argument("--prior-dim", type=int, default=256)


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


class LitPixelCNNTrainer(pl.LightningModule):
    num_codebook = 64
    dim_codebook = 32
    num_classes = 1
    prior_num_layers = 15
    vqvae_weights_path = "vqvae_cifar.pt"

    def __init__(self, lr=1e-3, prior_dim=512):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.vqvae = CifarVQVAE(
            num_codebook=self.num_codebook,
            dim_codebook=self.dim_codebook,
            codebook_flavor="classic",
        )
        self.vqvae.load_state_dict(torch.load(self.vqvae_weights_path))
        self.prior = GatedPixelCNN(
            input_dim=self.vqvae.codebook.num_codebook,
            dim=prior_dim,
            n_layers=self.prior_num_layers,
            n_classes=self.num_classes,
            extra_dim=False,
        )

    def training_step(self, images, batch_idx):
        encoding = self.vqvae.encode(images)
        # Switch to channel last
        encoding = encoding.permute(0, 2, 3, 1)
        quantized, indices, _ = self.vqvae.codebook.quantize(encoding)
        indices = indices.argmax(-1)
        batch_size = images.shape[0]
        labels = torch.zeros(
            batch_size, self.num_classes, device=images.device, dtype=int
        ).squeeze()
        logits = self.prior(indices, labels)
        loss = F.cross_entropy(logits, indices) / np.log(2)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        if batch_idx == 0:
            self._log_indice_reconstruction(
                sampled_indices=logits.argmax(1),
                quantized=quantized,
                name="reconstruction",
                subset="train",
            )
        return loss

    def validation_step(self, images, batch_idx):
        encoding = self.vqvae.encode(images)
        # Switch to channel last
        encoding = encoding.permute(0, 2, 3, 1)
        quantized, indices, _ = self.vqvae.codebook.quantize(encoding)
        indices = indices.argmax(-1)
        batch_size = images.shape[0]
        labels = torch.zeros(
            batch_size, self.num_classes, device=images.device, dtype=int
        ).squeeze()
        logits = self.prior(indices, labels)
        # Compute losses
        loss = F.cross_entropy(logits, indices) / np.log(2)
        self.log("val_loss", loss)
        patch = middle_patch_coordinates(indices.shape[1:], d=7)
        self._log_patched_nll(logits=logits, indices=indices, patch=patch)
        # Once an epoch, we log generated samples
        if batch_idx == 0:
            # Reconstructions
            self._log_indice_reconstruction(
                sampled_indices=logits.argmax(1),
                quantized=quantized,
                name="reconstruction",
                subset="val",
            )
            # Sampling
            indices_sampled = self.prior.generate(
                labels, batch_size=batch_size, shape=indices.shape[1:]
            )
            self._log_indice_reconstruction(
                sampled_indices=indices_sampled,
                quantized=quantized,
                name="sample",
                subset="val",
            )
            # Inpainting
            indices_inpainting = self.prior.generate_inpainting(
                observations=mask_patch(indices, patch),
                patch=patch,
                label=labels,
                shape=indices.shape[1:],
                batch_size=batch_size,
            )
            self._log_indice_reconstruction(
                sampled_indices=indices_inpainting,
                quantized=quantized,
                name="inpainting",
                subset="val",
                patch=patch,
            )

    def configure_optimizers(self):
        return torch.optim.Adam(self.prior.parameters(), lr=self.lr)

    def _log_patched_nll(self, logits, indices, patch):
        i_0, j_0, i_1, j_1 = patch
        nll_patched = F.cross_entropy(logits, indices, reduction="none") / np.log(2)
        nll_patched = nll_patched[..., i_0:i_1,j_0:j_1].mean()
        self.log("val_nll", nll_patched)

    def _log_indice_reconstruction(
        self, sampled_indices, quantized, name, subset, patch=None
    ):
        sampled_quantized = self.vqvae.codebook.codebook_lookup(sampled_indices)
        # Switch to channel first
        sampled_quantized = sampled_quantized.permute(0, 3, 1, 2)
        sampled_images = self.vqvae.decode(sampled_quantized)
        # Decode quantized
        quantized = quantized.permute(0, 3, 1, 2)
        if patch is not None:
            quantized = mask_patch(quantized, patch)
        recons_images = self.vqvae.decode(quantized)
        self.logger.experiment.track(
            Image(image_compare_reconstructions(recons_images, sampled_images)),
            name=name,
            epoch=self.current_epoch,
            context={"subset": subset},
        )


class Experiment:
    exp_name = "pcnn-cifar"
    dataset_path = "./datasets/cifar"
    litmodule = LitPixelCNNTrainer

    def __init__(self, args):
        # Load dataset
        self.datamodule = CifarDataModule(
            dataset_path=self.dataset_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        # Load LitModule
        if args.load_checkpoint is not None:
            self.litmodule = self.litmodule.load_from_checkpoint(args.load_checkpoint)
            print("Loaded litmodule.")
        else:
            self.litmodule = self.litmodule(lr=args.lr, prior_dim=args.prior_dim)

        # Load trainer
        logger = AimLogger(
            experiment=self.exp_name,
            system_tracking_interval=None,
            log_system_params=False,
        )
        logger.experiment["hparams"] = vars(args)
        print(f"Using logger {logger}.")
        checkpoint_callback = ModelCheckpoint(
            dirpath=Path("checkpoints") / self.exp_name,
            filename=f"{datetime.datetime.now().strftime('%Y_%m_%d__%H%M%S')}",
            monitor="val_loss",
            save_last=True,
        )
        self.trainer = pl.Trainer(
            max_epochs=args.epochs,
            gpus=args.gpus,
            logger=logger,
            callbacks=[checkpoint_callback],
            limit_val_batches=args.limit_val_batches,
        )


if __name__ == "__main__":
    args = parser.parse_args()
    exp = Experiment(args)

    exp.trainer.fit(exp.litmodule, datamodule=exp.datamodule)
    exp.trainer.validate(datamodule=exp.datamodule, ckpt_path="best")
