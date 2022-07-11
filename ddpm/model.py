import torch
import torch.nn as nn


class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Model.

    Abtract class for defining ddpm. In the following documentation, we refer to the
    number of total time steps as ``T``.

    Parameters
    ----------
    denoising_model:
        torch module allowing backpropagation.
    delta_schedule:
        schedule with shape ``(T,)``.
    num_steps:
        ``T``.
    parametrisation:
        What the diffusion model aims at sampling. One of ``"epsilon"``, ``"x_t"`` or
        ``"x_0"`` (see https://arxiv.org/pdf/2204.06125.pdf). Default is ``"epsilon"``.
    """

    def __init__(
        self,
        denoising_model: callable,
        delta_schedule: torch.Tensor,
        num_steps: int,
        parametrisation: str = "epsilon",
    ):
        super().__init__()
        self.denoising_model = denoising_model
        self.register_buffer("delta", delta_schedule)
        self._num_steps = num_steps
        self._parametrisation = parametrisation

    def posterior_sample(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Returns a sample from the posterior distribution.

        Samples under

        .. math::
            q(x_t|x_0).

        Parameters
        ----------
        x_0:
            Unaltered tensor with shape ``(batch_size, *)``.
        t:
            Target step of the noising process with shape ``(batch_size,)``.
        """
        raise NotImplementedError

    def prior_sample(self, shape: torch.Size) -> torch.Tensor:
        """Returns a sample from the prior distribution.

        Samples under

        .. math::
            p_\\theta(x_0) = \int p_\\theta(x_T) \prod_{s=T-1}^t p_\\theta(x_s|x_{s+1})

        Note
        ----
        The initial sample is drawn from the :py:meth:`prior_intial_sample` method.

        Parameters
        ----------
        shape:
            Shape of the sampled tensor.
        """
        x_t = self.prior_initial_sample(shape)
        time_steps = (
            torch.arange(1, self._num_steps, device=self.delta.device, dtype=torch.long)
            .unsqueeze(-1)
            .tile((1, shape[0]))
        )
        for t in reversed(time_steps):
            x_t = self.prior_onestep_sample(x=x_t, t=t)
        return x_t

    def prior_initial_sample(self, shape: torch.Size) -> torch.Tensor:
        """Returns a sample from the prior distribution at time ``T``.

        Samples under

        .. math::
            p_\\theta(x_T)

        Parameters
        ----------
        shape:
            Shape of the sampled tensor.
        """
        raise NotImplementedError

    def prior_onestep_sample(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Returns a sample from the denoising prior for a single step.

        Samples under

        .. math::
            p_\\theta(x_{t-1}|x_t)

        Parameters
        ----------
        x_t:
            Denoised vector at time ``t`` with shape ``(batch_size, *)``.
        t:
            Current step of the denoised vector ``x_t``, with shape ``(batch_size,)``.
        """
        raise NotImplementedError
