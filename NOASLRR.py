import numpy as np
import torch
import torch.nn as nn


def chebyshev_coeffs(f, K=10):
    B, N = f.shape
    x = torch.linspace(-1, 1, N, device=f.device)
    coeffs = []
    for k in range(K):
        T_k = torch.cos(k * torch.acos(x))
        c_k = (f * T_k).mean(dim=1) * 2
        coeffs.append(c_k)
    return torch.stack(coeffs, dim=1)

class LowRankLinear(nn.Module):
    def __init__(self, in_dim, out_dim, rank=3):
        super().__init__()
        self.A = nn.Linear(in_dim, rank, bias=False)
        self.B = nn.Linear(rank, out_dim, bias=True)
    def forward(self, x):
        return self.B(self.A(x))

class ResDeepONetLowRank(nn.Module):
    def __init__(self, branch_dim=[100,128,128], trunk_dim=[2,128,128],
                 K=10, lowrank_dim=128, lowrank_rank=3):
        super().__init__()
        self.K = K
        self.lowrank_dim = lowrank_dim

        layers = []
        for i in range(len(branch_dim)-1):
            layers.append(nn.Linear(branch_dim[i], branch_dim[i+1]))
            layers.append(nn.LayerNorm(branch_dim[i+1]))
            layers.append(nn.Tanh())
        self.branch = nn.Sequential(*layers)

        self.branch_spec = nn.Sequential(
            nn.Linear(K, branch_dim[-1]),
            nn.LayerNorm(branch_dim[-1]),
            nn.Tanh()
        )

        self.branch_lowrank = LowRankLinear(branch_dim[0], lowrank_dim, rank=lowrank_rank)
        self.lowrank_activation = nn.Tanh()

        self.gate_aux = nn.Sequential(
            nn.Linear(branch_dim[-1] + lowrank_dim, branch_dim[-1]),
            nn.Sigmoid()
        )

        self.gate_final = nn.Sequential(
            nn.Linear(branch_dim[-1]*2, branch_dim[-1]),
            nn.Sigmoid()
        )

        layers = []
        for i in range(len(trunk_dim)-1):
            layers.append(nn.Linear(trunk_dim[i], trunk_dim[i+1]))
            layers.append(nn.LayerNorm(trunk_dim[i+1]))
            layers.append(nn.Tanh())
        self.trunk = nn.Sequential(*layers)

    def forward(self, f, x):
        """
        f: (B, N) input functions
        x: (num_points, coord_dim) evaluation points
        """
        B, N = f.shape

        h_mlp = self.branch(f)

        h_spec = self.branch_spec(chebyshev_coeffs(f, K=self.K))

        h_low = self.lowrank_activation(self.branch_lowrank(f))

        alpha_aux = self.gate_aux(torch.cat([h_spec, h_low], dim=1))
        h_aux = alpha_aux * h_spec + (1 - alpha_aux) * h_low

        alpha_final = self.gate_final(torch.cat([h_mlp, h_aux], dim=1))
        h = alpha_final * h_mlp + (1 - alpha_final) * h_aux

        t = self.trunk(x)

        y = torch.einsum('bh,nh->bn', h, t)
        return y
