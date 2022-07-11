import torch
import torch.nn as nn

from vqvae.modules import ResNetBlock


class MiniimagenetAutoEncoder(nn.Module):
    """MiniImageNet AutoEncoder architecture based on ResNet blocks.

    Note
    ----
    In the following documentation, we will use the following variable names:
    `B`: batch size.
    `D`: number of channels of the feature map.

    Parameters
    ----------
    out_channels: number of channels of the feature map.
    """

    def __init__(self, out_channels):
        super().__init__()

        self.encode = nn.Sequential(
            ResNetBlock(in_channels=3, out_channels=16, stride=2),
            ResNetBlock(in_channels=16, out_channels=32, stride=2),
            ResNetBlock(in_channels=32, out_channels=32, stride=1),
            ResNetBlock(in_channels=32, out_channels=out_channels, stride=1),
        )

        self.decode = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=out_channels,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=3,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate the input tensor through the encoder and the decoder.

        Parameters
        ----------
        x: input tensor with shape `(B, 3, 88, 88)`.

        Returns
        -------
        Decoded representation with the same shape.
        """
        return self.decode(self.encode(x))
