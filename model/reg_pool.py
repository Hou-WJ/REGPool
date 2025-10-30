from typing import Optional, Tuple
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class SoftTopKRetain(nn.Module):
    def __init__(self, k: int, temperature: float = 0.1, hard=False):
        super().__init__()
        if not hard:
            self.k = k
        else:
            self.k = 1

        self.temperature = temperature
        assert k > 0, "k must be a positive integer"

    def forward(self, S: torch.Tensor) -> torch.Tensor:

        S_normalized = F.softmax(S, dim=-1)
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(S_normalized) + 1e-10) + 1e-10)
        S_perturbed = S_normalized + gumbel_noise

        topk_values, topk_indices = torch.topk(S_perturbed, k=self.k, dim=-1)
        mask = torch.zeros_like(S_normalized)
        mask.scatter_(-1, topk_indices, 1.0)

        softmax_weights = F.softmax(topk_values / self.temperature, dim=-1)
        expanded_weights = torch.zeros_like(S_normalized)
        expanded_weights.scatter_(-1, topk_indices, softmax_weights)

        S_hard = S_normalized * mask
        S_soft = S_normalized * expanded_weights
        S_topk = S_soft + (S_hard - S_soft).detach()

        return S_topk


def reg_pool(
        x: Tensor,
        adj: Tensor,
        s: Tensor,
        mask: Optional[Tensor] = None,
        normalize: bool = True,
        topk: SoftTopKRetain = None,
        temperature: float = 0.1,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    batch_size, num_nodes, _ = x.size()
    _, _, num_clusters = adj.size()

    if topk is not None:
        s = topk(s)
    else:
        s = torch.softmax(s / temperature, dim=-1)

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    link_loss = adj - torch.matmul(s, s.transpose(1, 2))
    link_loss = torch.norm(link_loss, p=2)
    if normalize is True:
        link_loss = link_loss / adj.numel()

    ent_loss = (-s * torch.log(s + 1e-15)).sum(dim=-1).mean()

    x_recon = torch.bmm(s, out)
    recon_loss = torch.mean((x - x_recon) ** 2)

    cluster_activation = s.sum(dim=1)
    prob = cluster_activation / (cluster_activation.sum(dim=-1, keepdim=True) + 1e-10)
    clu_loss = -torch.sum(prob * torch.log(prob + 1e-10), dim=-1).mean()

    return out, out_adj, s, link_loss, ent_loss, clu_loss, recon_loss
