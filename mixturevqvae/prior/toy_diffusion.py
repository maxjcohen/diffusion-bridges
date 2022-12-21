import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import aim
import numpy as np
import matplotlib.pyplot as plt

from ddpm.diffusion_bridge import DiffusionBridge


def swish(x):
    return x * torch.sigmoid(x)


class SphericalProjection(nn.Module):
    """Spherical VQ Embedding. Should be replaced by one in VQ later"""

    def __init__(self, K=10, m=2, init="random"):
        super(SphericalProjection, self).__init__()
        self.emb = nn.Embedding(K, m)
        self.K = K
        if init == "random":
            self.emb.weight.data.uniform_(-1.0 / K, 1.0 / K)
        else:
            with torch.no_grad():
                self.emb.weight.data = init
            self.emb.requires_grad = False
        self.temp = 40.0

    def forward(self, x):
        return F.normalize(self.emb(x), dim=-1)

    def straight_through(self, x, temp=None):
        hard = self.discretize(x, mode="sample_onehot", temp=temp)
        return torch.matmul(hard, self.emb.weight)

    def discretize(self, x, mode="deterministic", temp=None):
        if temp == None:
            temp = self.temp
        if mode == "sample_onehot" or mode == "sample_soft":
            embs = self.emb.weight
        else:
            embs = self.emb.weight.detach()
        e_j = F.normalize(embs, dim=-1)
        dot = x @ e_j.T
        # distance = 1 + ||x|| - 2
        dists = 1 + torch.norm(x, dim=-1).unsqueeze(-1) - 2 * dot
        if mode == "deterministic":
            return torch.argmin(dists, dim=-1)
        elif mode == "sample":
            return Categorical(logits=-dists * temp).sample()
        elif mode == "sample_onehot":
            return F.gumbel_softmax(-dists, tau=temp, dim=-1, hard=True)
        elif mode == "sample_soft":
            return F.gumbel_softmax(-dists, tau=temp, dim=-1, hard=False)

    def display(self, td):
        centroids = self.emb.weight.data.detach().numpy()
        print(f"norms of centroids: {np.linalg.norm(centroids, axis=-1)}")
        centroids = centroids / np.linalg.norm(centroids, axis=-1, keepdims=True)
        td.display_embedding(centroids)


class TimeEmbedding(nn.Module):
    """
    ### Embeddings for $t$
    """

    def __init__(self, n_channels: int):
        """
        * `n_channels` is the number of dimensions in the embedding
        """
        super().__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        # Create sinusoidal position embeddings
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with the MLP
        emb = swish(self.lin1(emb))
        emb = self.lin2(emb)

        return emb


class Net(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(Net, self).__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.embedding = TimeEmbedding(hidden_dim)
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, dim, 3, padding=1)

    def forward(self, x_t, t):
        x_p = self.linear1(x_t)
        shape = x_p.shape

        time_emb = self.embedding(t)
        time_emb = time_emb.view((shape[0],) + (1,) * (len(shape) - 2) + (shape[-1],))

        x = x_p + time_emb
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = torch.tanh(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        return x


class ToyModel(DiffusionBridge):
    theta = 2
    eta2 = 0.01
    diffusion_target = 0

    def __init__(
        self,
        denoising_model: callable,
        delta_schedule: torch.Tensor,
        num_steps: int,
        K,
        dim,
    ):
        super().__init__(
            denoising_model=denoising_model,
            delta_schedule=delta_schedule,
            num_steps=num_steps,
            theta=self.theta,
            eta2=self.eta2,
            diffusion_target=self.diffusion_target,
        )
        self.sp = SphericalProjection(K=K, m=dim)


class LitToyModel(pl.LightningModule):
    def __init__(
        self, denoising_model, delta_schedule, num_steps=20, K=8, dim=2, lr=1e-3
    ):
        super().__init__()
        self.save_hyperparameters("num_steps", "K", "dim", "lr")
        self.toy_model = ToyModel(
            denoising_model, delta_schedule, num_steps, K=K, dim=dim
        )

    def training_step(self, batch, batch_idx):
        loss = self.toy_model.compute_loss(batch)
        self.log("train_loss", loss)
        if batch_idx == 0:
            self.compute_elbo(batch)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    @torch.no_grad()
    def compute_elbo(self, encodings):
        def compute_Lt(encodings, t):
            batch_size = encodings.shape[0]
            # Reparametrize encodings_t
            noise = torch.randn_like(encodings)
            encodings_t = self.toy_model.posterior_sample(encodings, t, epsilon=noise)
            # Compute eps_theta
            eps_theta = self.toy_model.denoising_model(encodings_t, t)
            # Compute difference between noises
            diff = (
                F.mse_loss(noise, eps_theta, reduction="none")
                .flatten(start_dim=1)
                .sum(-1)
            )
            return diff

        time_steps = (
            torch.arange(
                2, self.hparams.num_steps, dtype=torch.long, device=encodings.device
            )
            .unsqueeze(-1)
            .tile(encodings.shape[0])
        )
        elbo_prior = torch.stack([compute_Lt(encodings, t) for t in time_steps])
        # Plot elbo_prior at time steps
        fig = plt.figure()
        plt.plot(time_steps[:, 0].cpu().numpy(), elbo_prior.mean(-1).cpu().numpy())
        plt.close(fig)
        self.logger.experiment.track(
            aim.Figure(fig), step=self.current_epoch, name="elbo prior"
        )
