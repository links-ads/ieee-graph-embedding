import torch
from torch import Tensor
import torch.nn as nn


class LogisticRegression(nn.Module):
    """Simple Logistic Regression Model: Linear + Sigmoid"""
    def __init__(self, in_size: int, device: str) -> None:
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(in_size, 1).to(device)
        self.sigma = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x).squeeze()

    def predict(self, x: Tensor):
        with torch.no_grad():
            out = self.forward(x)
        return self.sigma(out).round()


class NodeEmbedderNaive(nn.Module):
    """Model that simply returns input node embeddings, without any layer of computation"""
    def __init__(self) -> None:
        super(NodeEmbedderNaive, self).__init__()

    def forward(self, inps: Tensor, adjacency: Tensor):
        return inps


class LinearDecoder(nn.Module):
    """Simple Linear transformation.
    ATTENTION: It's not permutation invariant"""
    def __init__(self, in_size: int, out_size: int) -> None:
        super(LinearDecoder, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.linear = nn.Linear(in_size, out_size)

    def forward(self, inps: Tensor):
        return self.linear(inps)


class MaxPoolDecoder(nn.Module):
    """Simple max pool folowed by linear layer.
    This is permutation invariant"""
    def __init__(self, in_size: int, out_size: int) -> None:
        super(MaxPoolDecoder, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.linear = nn.Linear(in_size, out_size)

    def forward(self, inps: Tensor):
        """
        :param Tensor inps: batch of graph embeddings
                            of dim (batch_size, num_target_neighs, emb_size)
        """
        maxed_pooled = inps.max(dim=1).values
        return self.linear(maxed_pooled)
