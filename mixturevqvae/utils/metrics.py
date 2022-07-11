from typing import Tuple

import numpy as np
import torch
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm


def latent_usage(
        model: torch.nn.Module, dataloader: iter, device: torch.device = None
) -> float:
    """Compute ratio of used latent codes on this dataloader.

    Iterate over the dataloader and compute the model encding. Count the number of
    unique codes, compute the ratio with the model's `K`.

    Parameters
    ----------
    model: VQ model.
    dataloader: iterator of the dataset.
    device: torch device, default is `"cpu"`.

    Returns
    -------
    Usage ratio.
    """
    device = device or "cpu"

    latents = []
    with torch.no_grad():
        for batch, label in tqdm(dataloader):
            latents.extend(model.encode(batch.to(device)))
    latents = torch.stack(latents)

    return len(latents.unique()) / model.K


def reconstruction_cost(
        model: torch.nn.Module,
        dataloader: iter,
        device: torch.device = None,
) -> float:
    """Compute cost between original input and reconstruction by the model.

    Iterate over the dataloader and propagate inputs through the model. Reconstructions
    are compared to the originals through the criteria, currently the MSE.

    Parameters
    ----------
    model: VQ model.
    dataloader: iterator of the dataset.
    device: torch device, default is `"cpu"`.

    Returns
    -------
    Reconstruction cost.
    """
    device = device or "cpu"
    criteria = model.reconstruction_loss

    cost = 0
    with torch.no_grad():
        for batch, label in tqdm(dataloader):
            x_tilde = model(batch.to(device))[0]
            cost += criteria(batch, x_tilde)

    cost = cost / len(dataloader)
    return cost.item()


def bits_per_dim(
        model,
        dataloader: iter,
        device: torch.device = None,
) -> float:
    """Compute the bits per dimensions. See eg. https://arxiv.org/pdf/1601.06759.pdf

    bpd = NLL / dimensions / ln(2)

    Parameters
    ----------
    model: VQ model. Must have a log_prob function
    dataloader: iterator of the dataset.
    clamp: If `True`, model's reconstruction are clamped between `0` and `1`.
    Default is `True`.
    device: torch device, default is `"cpu"`.

    Returns
    -------
    Bits per dimensions (nats)
    """
    device = device or "cpu"
    dim = np.prod(next(iter(dataloader))[0].size()[1:])

    ll = 0
    with torch.no_grad():
        for batch, label in tqdm(dataloader):
            z = model.encode(batch)
            ll += model.log_prob(z, batch).sum()

    ll = ll / len(dataloader)
    return ll.item() / dim / np.log(2.)


def calculate_activation_statistics(images: torch.tensor, model: torch.nn.Module, device: str = "cpu") -> Tuple[
    float, float]:
    """
    Calculate mu and sigma of the model's activations given the input images
    Parameters
    ----------
    images : input images to get the stats from
    model : model to get the predictions from (Inceptionv3)
    device : torch device, default is `"cpu"`.

    Returns
    -------
    Tuple[float,float)
        (mean of the activations over the images, covariance)
    """
    model.eval()

    if device == "cpu":
        batch = images
    else:
        batch = images.cuda()
    pred = model(batch)[0]

    # If model output is not scalar, apply global spatial average pooling.
    # This happens if you choose a dimensionality not equal 2048.
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = torch.nn.functional.adaptive_avg_pool2d(pred, output_size=(1, 1))

    act = pred.cpu().data.numpy().reshape(pred.size(0), -1)

    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_frechet(images_real: torch.tensor, images_fake: torch.tensor, dims: int = 2048,
                      device: str = "cpu") -> float:
    """
    Call the library function to get the fid score with input images instead of images folder path
    Parameters
    ----------
    images_real : real image from the dataset
    images_fake : generated samples/images from the prior of the generative model
    dims : number of activations for the inception model
    device : torch device, default is `"cpu"`.

    Returns
    -------
    float:
        FID value
    """
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)

    mu_1, std_1 = calculate_activation_statistics(images_real, model, device=device)
    mu_2, std_2 = calculate_activation_statistics(images_fake, model, device=device)

    """get frechet distance"""
    fid_value = calculate_frechet_distance(mu_1, std_1, mu_2, std_2)
    return fid_value
