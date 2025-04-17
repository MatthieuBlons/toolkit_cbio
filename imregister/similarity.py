import torch
from torch import nn


class MutualInformation(nn.Module):

    def __init__(self, sigma=0.1, num_bins=256, normalize=True):
        super(MutualInformation, self).__init__()

        self.sigma = sigma
        self.num_bins = num_bins
        self.normalize = normalize
        self.epsilon = float(1e-10)
        self.bins = nn.Parameter(
            torch.linspace(0, 255, num_bins).float(), requires_grad=False
        )

    def _marginal(self, values):
        residuals = values.unsqueeze(-1) - self.bins  #
        kernel_values = torch.exp(-0.5 * (residuals / self.sigma).pow(2))
        p = torch.mean(kernel_values, dim=1)  # mean or sum ?
        normalization = torch.sum(p, dim=-1).unsqueeze(1) + self.epsilon
        p /= normalization
        return p, kernel_values

    def _joint(self, kernelx, kernely):
        joint_kernel_values = torch.matmul(kernelx.transpose(1, 2), kernely)
        normalization = joint_kernel_values.sum(dim=(1, 2), keepdim=True) + self.epsilon
        p = joint_kernel_values / normalization
        return p

    def _shan_entropy(self, p):
        B = p.shape[0]
        p_flat = p.flatten(start_dim=1)
        H = -torch.sum(p_flat * torch.log2(p_flat + self.epsilon), dim=-1)
        return H

    def _get_mutual_information(self, x, y):
        # for gray scale Torch tensors for images between (0, 1)
        assert x.shape == y.shape
        x = x * float(255)
        y = y * float(255)
        x_flat = x.flatten(start_dim=1)
        y_flat = y.flatten(start_dim=1)

        p_x, kernel_x = self._marginal(x_flat)
        p_y, kernel_y = self._marginal(y_flat)
        p_xy = self._joint(kernel_x, kernel_y)

        H_x = self._shan_entropy(p_x)
        H_y = self._shan_entropy(p_y)

        H_xy = self._shan_entropy(p_xy)
        mutual_information = H_x + H_y - H_xy
        if self.normalize:
            mutual_information = 2 * mutual_information / (H_x + H_y)

        return mutual_information

    def forward(self, x, y):
        return self._get_mutual_information(x, y)

    def _histogram(self, x):
        x = x * float(255)
        x_flat = x.flatten(start_dim=1)
        residuals = x_flat.unsqueeze(-1) - self.bins
        hist, _ = self._marginal(x_flat)
        return hist, residuals

    def _histogram2d(self, x, y):
        assert x.shape == y.shape
        x = x * float(255)
        y = y * float(255)
        x_flat = x.flatten(start_dim=1)
        y_flat = y.flatten(start_dim=1)

        _, kernel_x = self._marginal(x_flat)
        _, kernel_y = self._marginal(y_flat)
        hist2d = self._joint(kernel_x, kernel_y)
        return hist2d
