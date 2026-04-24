import torch
import torch.nn as nn


class DDPMScheduler(nn.Module):

    def __init__(self, num_time_steps: int = 1000):
        super().__init__()

        beta = torch.linspace(1e-4, 0.02, num_time_steps)
        alpha = 1.0 - beta
        alpha_hat = torch.cumprod(alpha, dim=0)

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_hat", alpha_hat)

    def forward(self, t):
        return (
            self.beta[t],
            self.alpha[t],
            self.alpha_hat[t],
        )