import os
import re
from tqdm import tqdm
import random
from typing import Dict, List, Tuple, Union
import json
import numpy as np
import datetime
from collections import deque
import torch
from torch import LongTensor
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


def split_graph_train_val(graph: Data, train_ratio: float) -> Tuple[Data, Data]:
    train_mask = torch.rand(graph.num_edges) < train_ratio
    edge_idx_train = to_undirected(graph.edge_index[:, train_mask])
    edge_idx_val = to_undirected(graph.edge_index[:, ~train_mask])
    graph_train = Data(init_emb=graph.init_emb, edge_index=edge_idx_train)
    graph_val = Data(init_emb=graph.init_emb, edge_index=edge_idx_val)
    return graph_train, graph_val


def get_neighbours(graph: Data, node: int) -> LongTensor:
    # Given a node of graph find its neighbours.
    mask = graph.edge_index[0] == node
    return graph.edge_index[1][mask]


def compute_furthest_nodes(graph: Data) -> Dict[int, int]:
    """Compute the furthest node for each node in the graph via BFS."""
    nx_graph = torch_geometric.utils.to_networkx(graph, to_undirected=True)
    furthest_nodes = {}
    for node in tqdm(nx_graph.nodes):
        visited = set()
        queue = deque([(node, 0)])
        visited.add(node)
        max_distance = 0
        set_of_furthest_nodes = set([])

        while queue:
            current_node, distance = queue.popleft()

            if distance > max_distance:
                max_distance = distance
                set_of_furthest_nodes = set([current_node])  # Reset set of furthest nodes.
            elif distance == max_distance:  # Add node to set of furthest nodes.
                set_of_furthest_nodes.add(current_node)

            for neighbor in nx_graph.neighbors(current_node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))

        furthest_nodes[node] = list(set_of_furthest_nodes)

    return furthest_nodes


def cos_sim_0(a: FloatTensor, b: FloatTensor) -> FloatTensor:
    a_norm = a / a.norm(p=2, dim=-1).unsqueeze(-1)
    b_norm = b / b.norm(p=2, dim=-1).unsqueeze(-1)
    return torch.mm(a_norm, b_norm.transpose(0, 1)).squeeze(0)


def cos_sim(a: FloatTensor, b: FloatTensor) -> FloatTensor:
    return F.cosine_similarity(a, b)


def batch_cos_sim(a: FloatTensor, b: FloatTensor) -> FloatTensor:
    """Compute batch cosine similarity along last dimension:
    Parameters
    ----------
    a: 3D FloatTensor (batch, n1, vec_size)
    b: 3D FloatTensor (batch, n2, vec_size)
    Returns
    -------
    batch_cos_sim: 3D FloatTensor (batch, n1, n2)"""
    bcossim = nn.CosineSimilarity(dim=2)
    return bcossim(a, b)


def dot_sim(a: FloatTensor, b: FloatTensor) -> FloatTensor:
    return torch.mm(a, b.transpose(0, 1)).squeeze(0)


def get_most_similar_users(user_emb: FloatTensor,
                           user2emb: Dict[str, FloatTensor],
                           N: int) -> List[Tuple[str, float]]:
    embs = torch.vstack(list(user2emb.values()))
    sims = cos_sim(user_emb.unsqueeze(0), embs)
    users = list(user2emb.keys())
    idxs = torch.argsort(sims, descending=True)
    res = [(users[idx], sims[idx]) for idx in idxs[: N]]
    return res


def normalize_adjacency(adj: torch.FloatTensor):
    """Symmetrically normalize adjacency matrix following https://arxiv.org/abs/1609.02907:
    adj_norm = D^-1/2 * (adj + I) * D^-1/2,  where I is the identity and D is the
    degree matrix adj.sum(-1).
    Intuitively the connection between u and v is normalized with their respective degrees:
    conn_norm(u, v) = conn(u, v) / sqrt(deg(v) * deg(u))
    Parameters
    -----------
    adj: 2D FloatTensor (N_nodes, N_nodes)

    Returns
    -------
    adj_norm: 2D FloatTensor
    """
    # Add self-edges if diagonal is zero.
    if (adj.diag() == 0).all():
        adj += np.identity(adj.shape[0])
    # Normalize with degree matrix.
    D = torch.diag(adj.sum(dim=-1))
    D_inv_sqrt = torch.pow(D, -0.5)
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.
    return torch.mm(torch.mm(D_inv_sqrt, adj), D_inv_sqrt)


class FileIO:
    @staticmethod
    def read_text(filename):
        with open(filename, "r", encoding="utf8") as f:
            return f.read()

    @staticmethod
    def write_text(data, filename):
        with open(filename, "w", encoding="utf8") as f:
            f.write(data)

    @staticmethod
    def append_text(data: str, filename):
        with open(filename, "a", encoding="utf8") as f:
            f.write("\n" + data)

    @staticmethod
    def read_json(filename):
        with open(filename, "r", encoding="utf8") as f:
            return json.load(f)

    @staticmethod
    def write_json(data, filename):
        with open(filename, "w", encoding="utf8") as f:
            json.dump(data, f)

    @staticmethod
    def write_numpy(filename: str, array: np.array) -> None:
        with open(filename, 'wb') as f:
            np.save(f, array)

    @staticmethod
    def read_numpy(filename: str) -> None:
        with open(filename, 'rb') as f:
            return np.load(f)


def extract_nums_from_string(string: str) -> List[Union[int, float]]:
    return [
        int(x) if x.isdigit() else float(x)
        for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)", string)
    ]


def current_timestamp() -> str:
    return datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M")


def set_savefolder(output_folder: str, model_type: str, run_name: str) -> str:
    dirname = f"{output_folder}/{model_type}/{run_name}"
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return dirname


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
