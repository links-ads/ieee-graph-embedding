from typing import Any
import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.utils import softmax, add_self_loops
from torch_geometric.nn.dense.linear import Linear


class MyGATConv(MessagePassing):
    """
    Each head gets embeddings of size 'emb_size'/'n_heads', where attention is computed
    independently. The embeddings are then concatenated at the
    end of the GAT layer into a single embedding of size 'emb_size'.
    """
    def __init__(self, in_size: int, emb_size: int, n_heads: int, dropout: float,
                 unitary: bool, concat: bool = True,
                 leaky_relu_slope: float = 0.2) -> None:
        super(MyGATConv, self).__init__(aggr='add', node_dim=0)
        assert emb_size % n_heads == 0, "GAT layer 'emb_size' should be a multiple of n_heads due to concatenation"
        self.emb_size_per_head = emb_size // n_heads  # Size of embeddings for each separate head of the attention.
        self.n_heads = n_heads
        self.emb_size = emb_size
        self.concat = concat
        self.unitary = unitary
        self.leaky_relu_slope = leaky_relu_slope
        self.dropout = dropout
        self.W_i = Linear(in_size, emb_size, weight_initializer='glorot')
        self.W_j = Linear(in_size, emb_size, weight_initializer='glorot')

        self.W_i.reset_parameters()
        self.W_j.reset_parameters()

        self.att = nn.Parameter(Tensor(n_heads, 1 * self.emb_size_per_head, 1))
        # self.att = nn.Parameter(Tensor(1, n_heads, self.emb_size_per_head))
        glorot(self.att)

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        edge_index, _ = add_self_loops(edge_index)
        H, C = self.n_heads, self.emb_size_per_head
        x_i = self.W_i(x).view(-1, H, C)
        x_j = self.W_j(x).view(-1, H, C)
        out = self.propagate(edge_index=edge_index, x=(x_i, x_j))
        if self.concat:
            out = out.view(-1, self.emb_size)
        else:
            out = out.mean(dim=1)  # Average over attention heads.

        if self.unitary:  # Map to unitary sphere.
            out = out / (out.norm(dim=-1) + 1e-9).unsqueeze(-1)
        return out

    def message(self, x_j: Tensor, x_i: Tensor, index: Tensor) -> Tensor:
        """
        x_j: Tensor = Keys
        x_i: Tensor = Querys

        Returns:
        o_j = alpha_j * x_j (used in self.propagate: h_i = sum_j o_j = sum_j alpha_ij x_j)
        """
        x = torch.cat((x_i, x_j), dim=-1)
        x = x_i + x_j
        x = F.leaky_relu(x, self.leaky_relu_slope)
        # Compute Multi-head Attention via batch matrix mult between att and x:
        # att * x = (heads, num_nodes, emb_size) * (heads, emb_size, 1) = (heads, num_nodes, 1)
        # Put final output in the form (num_nodes, heads, 1)
        alpha = torch.bmm(x.transpose(0, 1), self.att).transpose(0, 1)
        alpha = softmax(alpha, index, dim=0)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha


def glorot(value: Any) -> None:
    if isinstance(value, Tensor):
        stdv = math.sqrt(6.0 / (value.size(-2) + value.size(-1)))
        value.data.uniform_(-stdv, stdv)
    else:
        for v in value.parameters() if hasattr(value, 'parameters') else []:
            glorot(v)
        for v in value.buffers() if hasattr(value, 'buffers') else []:
            glorot(v)
