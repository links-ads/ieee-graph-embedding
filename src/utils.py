import os
import re
import logging
from tqdm import tqdm
import random
from typing import Dict, List, Tuple, Union
import json
import numpy as np
import datetime
from collections import deque
import networkx as nx
import torch
from torch import LongTensor
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import (to_undirected, subgraph, remove_isolated_nodes,
                                   to_dense_adj)

logger = logging.getLogger('graphembedding')


def split_graph_edges_train_val(graph: Data, train_ratio: float) -> Tuple[Data, Data]:
    '''Split edges for training and validation'''
    train_mask = torch.rand(graph.num_edges) < train_ratio
    edge_idx_train = to_undirected(graph.edge_index[:, train_mask])
    edge_idx_val = to_undirected(graph.edge_index[:, ~train_mask])
    graph_train = Data(x=graph.x, edge_index=edge_idx_train)
    graph_val = Data(x=graph.x, edge_index=edge_idx_val)
    return graph_train, graph_val


def split_graph_nodes_train_val(graph: Data, train_ratio: float) -> Tuple[Data, Data]:
    '''Split nodes for training and validation'''
    num_nodes = graph.num_nodes
    num_train = int(train_ratio * num_nodes)
    perm = torch.randperm(num_nodes)
    train_idx = perm[:num_train]
    val_idx = perm[num_train:]
    # Create training and validation edge_indexes.
    edge_idx_train, _ = subgraph(train_idx, graph.edge_index, relabel_nodes=True)
    edge_idx_train, _, mask = remove_isolated_nodes(edge_idx_train)
    train_idx = train_idx[mask]  # Remove isolated nodes from train_idx.
    edge_idx_train = make_undirected(edge_idx_train)
    edge_idx_val, _ = subgraph(val_idx, graph.edge_index, relabel_nodes=True)
    edge_idx_val, _, mask = remove_isolated_nodes(edge_idx_val)
    val_idx = val_idx[mask]  # Remove isolated nodes from val_idx.
    edge_idx_val = make_undirected(edge_idx_val)
    # Create graph_training and graph_validation.
    graph_training = Data(x=graph.x[train_idx], edge_index=edge_idx_train,
                          y=graph.y[train_idx])
    graph_validation = Data(x=graph.x[val_idx], edge_index=edge_idx_val,
                            y=graph.y[val_idx])

    return graph_training, graph_validation


def make_undirected(edge_index: LongTensor) -> LongTensor:
    '''
    Make the graph undirected.
    torch-geometric.utils.to_undirected() do not work properly.

    :param edge_index: 2D LongTensor representing the edge_index.
    :return: 2D LongTensor representing the edge_index of the undirected graph.
    '''
    # For each edge add edge in the opposite direction.
    row, col = edge_index[0], edge_index[1]
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)
    # Remove duplicate edges.
    edge_index = torch.unique(edge_index, dim=1)
    # Sort by row.
    rows_sorted_idx = torch.argsort(edge_index)[0]
    return edge_index[:, rows_sorted_idx]


def check_is_undirected(edge_index: LongTensor) -> bool:
    '''
    Check if the graph is undirected.
    torch_geometric.utils.is_undirected() do not work properly.

    :param edge_index: 2D LongTensor representing the edge_index.
    :return: True if the graph is undirected, False otherwise.
    '''
    n_edges = edge_index.shape[1]
    already_seen = set([])
    for i in tqdm(range(n_edges)):
        x = edge_index[0, i]
        y = edge_index[1, i]
        if x > y:
            x, y = y, x
        edge_name = f'{x}-{y}'
        if edge_name in already_seen:
            already_seen.remove(edge_name)
        else:
            already_seen.add(edge_name)
    return len(already_seen) == 0


def check_self_loops(edge_index: LongTensor) -> bool:
    '''Check if there are self loops in the edge_index.'''
    return (edge_index[0] == edge_index[1]).any()


def get_neighbours(graph: Data, node: int) -> LongTensor:
    # Given a node of graph find its neighbours.
    mask = graph.edge_index[0] == node
    return graph.edge_index[1][mask]


def compute_topn_far_nodes(graph: Data, N: int) -> Dict[int, List[int]]:
    '''Compute the N nodes that are the furthest from each node in the graph.

    :param graph: PyG Data object
    :param N: Number of furthest nodes to compute
    :return: Dictionary with node as key and list of furthest nodes as value
    '''
    # Convert edge_index to an adjacency matrix.
    adj = to_dense_adj(graph.edge_index)[0]
    # Convert the adjacency matrix to a NetworkX graph for easier distance computation.
    G = nx.from_numpy_array(adj.numpy())
    # Store furthest nodes for each node.
    furthest_nodes = {}
    for node in tqdm(G.nodes(), desc='Computing furthest nodes', delay=20):
        # Compute shortest path lengths from node to all other nodes.
        lenghts = nx.single_source_shortest_path_length(G, node)
        # Sort the nodes by distance in descending order.
        sorted_trgs_dist = sorted(lenghts.items(), key=lambda x: x[1], reverse=True)
        # Store the N furthest nodes.
        furthest_nodes[node] = sorted_trgs_dist[: N]
    return furthest_nodes


def compute_furthest_nodes(graph: Data) -> Dict[int, Dict[int, int]]:
    """Compute the furthest node for each node in the graph via BFS."""
    nx_graph = torch_geometric.utils.to_networkx(graph)
    furthest_nodes = {}
    for node in tqdm(nx_graph.nodes):
        visited = set()
        queue = deque([(node, 0)])
        visited.add(node)
        max_distance = 0
        node_furthest_nodes = {}
        # BFS.
        while queue:
            current_node, distance = queue.popleft()
            if distance > max_distance:
                max_distance = distance
                node_furthest_nodes = {int(current_node): distance}  # Reset dict of furthest nodes.
            elif distance == max_distance:  # Add node to dict of furthest nodes.
                node_furthest_nodes[int(current_node)] = distance
            # Add unvisited neighbors to queue.
            for neighbor in nx_graph.neighbors(current_node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))
        furthest_nodes[int(node)] = node_furthest_nodes
    return furthest_nodes


def get_neighs_and_furthest_nodes(graph_uu: Data,
                                  compute: bool,
                                  file_neighs: str,
                                  file_furthest: str) -> Tuple[dict, dict]:
    if compute:
        logger.info('Computing neighbour nodes')
        neighbour_nodes = {
            int(node_idx): get_neighbours(graph_uu, node_idx).tolist()
            for node_idx in tqdm(range(graph_uu.num_nodes))
        }
        logger.info('Computing furthest nodes')
        furthest_nodes = compute_topn_far_nodes(graph_uu)
        FileIO.write_json(neighbour_nodes, file_neighs)
        FileIO.write_json(furthest_nodes, file_furthest)
    else:
        neighbour_nodes = FileIO.read_json(file_neighs)
        furthest_nodes = FileIO.read_json(file_furthest)
    return neighbour_nodes, furthest_nodes


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
