from pathlib import Path

import torch
import torchvision

from ..utils import parser
from .cifar import LitHo, Experiment

parser.set_defaults(batch_size=64)


class LitHoSampleGenerator(LitHo):
    root = Path("generated") / "ho" / "cifar"
    image_export_path = root / "images"
    image_export_path.mkdir(parents=True, exist_ok=True)

    def validation_step(self, batch, batch_idx):
        encodings = self.vqvae.encode(batch)
        x = self.sample_latent(torch.randn_like(encodings))
        zq, indices, _ = self.vqvae.codebook.quantize(x.permute(0, 2, 3, 1))
        sampled_images = self.vqvae.decode(zq.permute(0, 3, 1, 2)).clip(0, 1)
        indices = indices.argmax(-1)
        for idx, image in enumerate(sampled_images):
            torchvision.transforms.functional.to_pil_image(image).save(
                self.image_export_path.joinpath(f"{batch_idx}_{idx}.jpg")
            )
        return indices

    def validation_epoch_end(self, outputs):
        indices = torch.cat(outputs, dim=0)
        torch.save(indices, self.root.joinpath("indices.pt"))

    @torch.no_grad()
    def sample_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Sample latent.

        Iteratively sample from $p_\theta(x_{t-1}|x_t)$ and quantize.

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
        for t in reversed(time_steps):
            x = self.diffusion_model.p_sample(xt=x, t=t)
        return x


Experiment.exp_name = "ho-cifar-export"
Experiment.litmodule = LitHoSampleGenerator


if __name__ == "__main__":
    args = parser.parse_args()
    exp = Experiment(args)
    exp.trainer.validate(exp.litmodule, datamodule=exp.datamodule)
