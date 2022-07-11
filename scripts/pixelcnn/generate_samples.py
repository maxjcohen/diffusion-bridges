from pathlib import Path

import torch
import torchvision

from ..utils import parser
from .train import LitPixelCNNTrainer, Experiment

parser.set_defaults(batch_size=64)


class LitPixelCNNSampleGenerator(LitPixelCNNTrainer):
    root = Path("generated") / "pcnn" / "cifar"
    image_export_path = root / "images"
    image_export_path.mkdir(parents=True, exist_ok=True)
    generate_map_shape = (16, 16)

    def validation_step(self, images, batch_idx):
        batch_size = images.shape[0]
        labels = torch.zeros(
            batch_size, self.num_classes, device=images.device, dtype=int
        ).squeeze()
        # Sampling
        indices_sampled = self.prior.generate(
            labels, batch_size=batch_size, shape=self.generate_map_shape
        )
        # Export images
        sampled_quantized = self.vqvae.codebook.codebook_lookup(indices_sampled)
        sampled_quantized = sampled_quantized.permute(0, 3, 1, 2)
        sampled_images = self.vqvae.decode(sampled_quantized).clip(0, 1)
        for idx, image in enumerate(sampled_images):
            torchvision.transforms.functional.to_pil_image(image).save(
                self.image_export_path.joinpath(f"{batch_idx}_{idx}.jpg")
            )
        return indices_sampled

    def validation_epoch_end(self, outputs):
        indices = torch.cat(outputs, dim=0)
        torch.save(indices, self.root.joinpath("indices.pt"))


Experiment.exp_name = "pcnn-cifar-export"
Experiment.litmodule = LitPixelCNNSampleGenerator

if __name__ == "__main__":
    args = parser.parse_args()
    exp = Experiment(args)
    exp.trainer.validate(exp.litmodule, datamodule=exp.datamodule)
