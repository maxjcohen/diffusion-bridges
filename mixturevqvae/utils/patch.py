from typing import List

import torch


def middle_patch_coordinates(shape: List[int], d: int) -> List[int]:
    """Compute coordinates for patch in the middle of a given shape.

    Parameters
    ----------
    shape: shape of the original map.
    d: diameter of the patch.

    Warning
    -------
    Both `shape` and `d` must be odd.

    Returns
    -------
    A list of coordinates `(i_0, j_0, i_1, j_1)`.
    """
    return (
        shape[0] // 2 - d // 2,
        shape[1] // 2 - d // 2,
        shape[0] // 2 + d // 2,
        shape[1] // 2 + d // 2,
    )


def mask_patch(tensor: torch.Tensor, patch_coordinates: List[int]) -> torch.Tensor:
    """Mask a tensor with a given patch.

    Parameters
    ----------
    tensor: tensor to mask.
    patch_coordinates: coordinates `(i_0, j_0, i_1, j_1)`.

    Returns
    -------
    Masked tensor.
    """
    i_0, j_0, i_1, j_1 = patch_coordinates
    mask = torch.zeros(tensor.shape, device=tensor.device, dtype=bool)
    mask[..., i_0:i_1, j_0:j_1] = True
    return tensor.masked_fill(mask, 0)
