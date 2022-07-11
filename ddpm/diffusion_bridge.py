from functools import wraps

import torch
import torch.nn.functional as F
import numpy as np

from .model import DDPM


def gaussian_sampler(gaussian_distribution):
    @wraps(gaussian_distribution)
    def sampler(*args, epsilon=None, **kwargs):
        mean, var = gaussian_distribution(*args, **kwargs)
        if not isinstance(var, torch.Tensor):
            var = torch.tensor(var)
        epsilon = epsilon if epsilon is not None else torch.randn_like(mean)
        return mean + var.sqrt() * epsilon

    return sampler


def unsqueeze(t, x):
    return t.reshape(t.shape + (1,) * len(x.shape[1:]))


class DiffusionBridge(DDPM):
    """Our proposed model.

    In the following documentation, :math:`x_*` refers to the diffusion target, ``T`` is
    the number of time steps. When ``t`` is an argument to a method, it always refers to
    the *target* time step of the process.

    Note
    ----
    In the implementation, the exponential forms are usually replaced by :math:`\\beta`
    or :math:`\\alpha`, with:

    .. math::
       \\beta_t = 1 - e^{-2 \\vartheta \delta_t}

       \\alpha_t = 1 - \\beta_t = e^{-2 \\vartheta \delta_t}

       \\bar \\alpha_t = \prod_{s=1}^t \\alpha_s = e^{-2 \\vartheta \sum_{s=1}^t
       \delta_s}
    """

    def __init__(
        self,
        denoising_model: callable,
        delta_schedule: torch.Tensor,
        num_steps: int,
        theta,
        eta2,
        diffusion_target,
    ):
        super().__init__(
            denoising_model=denoising_model,
            delta_schedule=delta_schedule,
            num_steps=num_steps,
        )
        self.theta = theta
        self.eta2 = eta2
        self.diffusion_target = diffusion_target
        self.register_buffer("beta", 1 - torch.exp(-2 * self.theta * delta_schedule))
        self.register_buffer("alpha", 1 - self.beta)
        self.register_buffer("alpha_bar", torch.cumprod(self.alpha, dim=0))
        self.register_buffer("sigma2", self.beta)

    @gaussian_sampler
    def posterior_sample(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """Returns a sample from the posterior distribution.

        Samples under

        .. math::
            q(x_t|x_0) = \mathcal N (x_* + (x_0 - x_*)\sqrt{\\bar \\alpha_t},
            \\frac{\eta^2}{2 \\vartheta}(1 - \\bar \\alpha_t)).

            = \mathcal N (x_* + (x_0 - x_*)e^{-\\vartheta \sum_{s=1}^t \delta_s},
            \\frac{\eta^2}{2 \\vartheta}(1 - e^{-2 \\vartheta \sum_{s=1}^t \delta_s}).

        Parameters
        ----------
        x:
            Unaltered tensor :math:`x_0` with shape ``(batch_size, *)``.
        t:
            Target step of the noising process with shape ``(batch_size,)``. From ``1``
            to ``T`` included.
        epsilon: Optional :py:class:`torch.Tensor`
            Fix the noise vector used for sampling.
        """
        t = unsqueeze(t, x) - 1
        mean = (
            self.diffusion_target
            + (x - self.diffusion_target) * self.alpha_bar[t].sqrt()
        )
        var = self.eta2 / (2 * self.theta) * (1 - self.alpha_bar[t])
        return mean, var

    @gaussian_sampler
    def posterior_onestep_sample(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Returns a sample from the posterior after a single noising step.

        Samples under

        .. math::
            q(x_t|x_{t-1}) = \mathcal N (x_* +(x_{t-1} - x_*)\sqrt{\\alpha_t},
            \\frac{\eta^2}{2 \\vartheta} \\beta_t).

            = \mathcal N (x_* +(x_{t-1} - x_*)e^{-\\vartheta \delta_t},
            \\frac{\eta^2}{2 \\vartheta} e^{-2 \\vartheta \delta_t}).

        Parameters
        ----------
        x:
            Noised vector :math:`x_{t-1}` with shape ``(batch_size, *)``.
        t:
            Target step of the noising process with shape ``(batch_size,)``.  From ``1``
            to ``T`` included.
        epsilon: Optional :py:class:`torch.Tensor`
            Fix the noise vector used for sampling.
        """
        t = unsqueeze(t, x) - 1
        mean = (
            self.diffusion_target + (x - self.diffusion_target) * self.alpha[t].sqrt()
        )
        var = self.eta2 / (2 * self.theta) * self.beta[t]
        return mean, var

    @gaussian_sampler
    def prior_initial_sample(self, shape: torch.Tensor) -> torch.Tensor:
        """Returns a sample from the prior distribution at time ``T``.

        Samples under

        .. math::
            p_\\theta(x_T) = \mathcal N (x_*, \\frac{\eta^2}{2 \\vartheta})

        Parameters
        ----------
        shape:
            Shape of the sampled tensor.
        """
        diffusion_target = self.diffusion_target
        if not isinstance(diffusion_target, torch.Tensor):
            diffusion_target = torch.full(
                shape, diffusion_target, dtype=torch.float, device=self.delta.device
            )
        return diffusion_target, self.eta2 / (2 * self.theta)

    @gaussian_sampler
    def prior_onestep_sample(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Returns a sample from the denoising prior for a single step.

        Samples under

        .. math::
            p_\\theta(x_t|x_{t+1}) = \mathcal N \left( \\frac{1}{\sqrt{\\alpha_{t+1}}}
            \left(x_{t+1} - \sqrt{\\frac{\eta^2}{2 \\vartheta ( 1 - \\bar \\alpha_{t+1}
            )}} \\beta_{t+1} \epsilon_\\theta(x_{t+1}, t+1) \\right), \\beta_{t+1}
            \\right)

        Parameters
        ----------
        x:
            Denoised vector :math:`x_{t+1}` with shape ``(batch_size, *)``.
        t:
            Target step of the denoised vector :math:`x_t`, with shape ``(batch_size,)``.
            From ``0`` to ``T-1`` included.
        """
        epsilon = self.denoising_model(x, t)
        t = unsqueeze(t, x)
        mean = (
            1
            / self.alpha[t].sqrt()
            * (
                x
                - self.beta[t]
                * torch.sqrt(self.eta2 / (2 * self.theta * (1 - self.alpha_bar[t])))
                * epsilon
            )
        )
        var = self.eta2 / (2 * self.theta) * self.sigma2[t]
        return mean, var

    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute diffusion loss term.

        Parameters
        ----------
        x:
            Unaltered tensor with shape ``(batch_size, *)``.

        Note
        ----
            Samples ``t`` uniformely between ``1`` and ``T-1`` included.
        """
        # Sample t
        t = torch.randint(
            1, self._num_steps, (x.shape[0],), device=x.device, dtype=torch.long
        )
        # Sample x_t
        epsilon_sampled = torch.randn_like(x)
        x_t = self.posterior_sample(x=x, t=t, epsilon=epsilon_sampled)
        # Compute epsilon_theta
        epsilon_theta = self.denoising_model(x_t, t)
        return F.mse_loss(epsilon_sampled, epsilon_theta)
