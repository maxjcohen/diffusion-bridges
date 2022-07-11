import argparse

parser = argparse.ArgumentParser(description="diffusion bridge helper script.")
parser.add_argument(
    "--epochs",
    default=100,
    type=int,
    help="Number of epochs for training. Default is 100.",
)
parser.add_argument(
    "--num-workers",
    default=4,
    type=int,
    help="Number of workers for dataloaders. Default is 1.",
)
parser.add_argument(
    "--batch-size",
    default=16,
    type=int,
    help="Batch size for dataloaders. Default is 16.",
)
parser.add_argument(
    "--gpus", default="0", type=str, help="Number of gpus. Default is `0`"
)
parser.add_argument("--lr", default=None, type=float, help="Learning rate.")
parser.add_argument(
    "--limit-val-batches",
    default=None,
    type=int,
    help="Maximum number of validation batches to evaluate on.",
)

# Weights
parser.add_argument(
    "--vqvae-weights", default=None, type=str, help="Path to vqvae weights."
)
parser.add_argument(
    "--load-checkpoint",
    default=None,
    type=str,
    help="Pytorch lightning checkpoint path to load from.",
)
