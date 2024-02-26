import os
import logging
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GAT
from torch_geometric.typing import Adj
from torch_geometric.utils import add_self_loops
from src.models.gat.layers import MyGATConv
from src.utils import FileIO

logger = logging.getLogger('graphembedding')


class GATstandard(nn.Module):
    def __init__(self,
                 in_size: int,
                 hidden_size: int,
                 num_layers: int,
                 out_size: int,
                 dropout: float,
                 device: str) -> None:
        """Simple wrapper around the torch_gemoetric.GAT class,
        which implements 'save' and 'load' methods."""
        super(GATstandard, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_size = out_size
        self.dropout = dropout
        self.model = GAT(in_channels=in_size, hidden_channels=hidden_size,
                         num_layers=num_layers, out_channels=out_size,
                         dropout=dropout, v2=True).to(device)

    def forward(self, init_embs: Tensor, adjacency: Tensor) -> Tensor:
        return self.model(init_embs, adjacency)

    def save(self, folder: str):
        model_folder = f'{folder}/GAT_in{self.in_size}_h{self.hidden_size}_nlayers{self.num_layers}_out{self.out_size}'
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
        model_params = {
            'in_size': self.in_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'out_size': self.out_size,
            'dropout': self.dropout
        }
        FileIO.write_json(model_params, f'{model_folder}/model_params.json')
        torch.save(self.model.state_dict(), f'{model_folder}/model_state_dict.json')

    @classmethod
    def load(cls, folder, device: str):
        model_params = FileIO.read_json(f'{folder}/model_params.json')
        mygat = cls(**model_params, device=device)
        mygat.model.load_state_dict(torch.load(f'{folder}/model_state_dict.json'))
        mygat.model.eval()
        return mygat


class samGAT(nn.Module):
    def __init__(self,
                 in_size: int,
                 emb_size: int,
                 n_heads: int,
                 out_size: int,
                 num_layers: int,
                 unitary: bool,
                 dropout: float) -> None:
        super(samGAT, self).__init__()
        assert emb_size % n_heads == 0, "Embedding size has to be multiple of n_heads \
            (due to the implementation of multi-head attention)"
        self.emb_size = emb_size
        self.num_layers = num_layers
        # The first layer has input_size = initial_node_feature_size
        # All other layers have input_size = output_size = emb_size.
        self.gat_layers = nn.ModuleList([
            MyGATConv(in_size, emb_size, n_heads, dropout, unitary)
        ])
        for _ in range(num_layers - 1):
            self.gat_layers.append(
                MyGATConv(emb_size, emb_size, n_heads, dropout, unitary),
            )
        self.out = nn.Linear(emb_size, out_size)

    def forward(self, x: Tensor, edge_index: Adj):
        edge_index, _ = add_self_loops(edge_index)
        for layer in self.gat_layers:
            x = layer(x, edge_index)
            x = F.leaky_relu(x, negative_slope=0.2)
        return self.out(x)

    def save(self, dirname: str) -> None:
        torch.save(self, f'{dirname}/model.pth')
        logger.info(f"Saved model to: {dirname}/model.pth")

    @classmethod
    def from_saved(cls, dirname: str, device: str) -> None:
        logger.info(f"Loading model form: {dirname}/model.pth")
        model = torch.load(f'{dirname}/model.pth', map_location=device)
        model.eval()
        return model
