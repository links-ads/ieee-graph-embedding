from typing import Dict, List, Tuple
import json
import openpyxl
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor


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
    def read_excel(filename, sheet_name="Sheet1"):
        wb_obj = openpyxl.load_workbook(filename)
        return wb_obj[sheet_name]

    @staticmethod
    def read_csv(filename: str) -> List[Dict]:
        with open(filename, "r", encoding="utf8") as f:
            csvreader = csv.reader(f)
            header = next(csvreader)
            return [
                {h: x for h, x in zip(header, row) if h}
                for row in csvreader
            ]

    @staticmethod
    def write_numpy(filename: str, array: np.array) -> None:
        with open(filename, 'wb') as f:
            np.save(f, array)

    @staticmethod
    def read_numpy(filename: str) -> None:
        with open(filename, 'rb') as f:
            return np.load(f)
