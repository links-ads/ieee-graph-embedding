from typing import Tuple, List, Dict
from tqdm import tqdm
import numpy as np
import networkx as nx
import torch
from torch import Tensor
from torch_geometric.utils import to_undirected, add_self_loops
from torch_geometric.data import Data
from src.models.graph_att_nets import myGAT
from src.utils import FileIO


class Archetypes:
    EMB_SIZE = 768

    def __init__(self,
                 user2tweets_file: str,
                 user2neigs_file: str,
                 user2init_emb_file: str,
                 user2graph_emb_file: str) -> None:
        self.user2tweets_file = user2tweets_file
        self.user2neigs_file = user2neigs_file
        self.user2init_emb_file = user2init_emb_file
        self.user2graph_emb_file = user2graph_emb_file

    def compute_initial_embeddings(self, encoder):
        """Compute initial user embeddings as the average of user's tweets
        text encoding"""
        user2tweets = FileIO.read_json(self.user2tweets_file)
        user2embs = {}
        for u, tweets in tqdm(user2tweets.items()):
            if tweets:
                try:
                    user2embs[u] = encoder.encode(tweets).mean(0).tolist()
                except Exception:
                    continue
        FileIO.write_json(user2embs, self.user2init_emb_file)

    def get_initial_embeddings(self) -> Dict[str, np.array]:
        """Initial Embeddings are the average of user tweets encoding"""
        return list(FileIO.read_json(self.user2init_emb_file).values())

    def get_graph_embeddings(self) -> Dict[str, np.array]:
        """Graph Embeddings, computed after applying GAT to initial embeddigns"""
        return list(FileIO.read_json(self.user2graph_emb_file).values())

    def get_adjacency_and_features(self) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        # Get users that have tweets.
        user2tweets = FileIO.read_json(self.user2tweets_file)
        nonzero_users = set([u for u, tweets in user2tweets.items() if tweets])
        # Do not consider archetypes or neighbours that do not have tweets.
        user2emb = {
            u: e for u, e in FileIO.read_json(self.user2init_emb_file).items()
            if u in nonzero_users
        }
        # Only consider archetypes and connections that have an initial embedding.
        archtype2conn = {
            u: [n for n in neighs if n in user2emb]
            for u, neighs in FileIO.read_json(self.user2neigs_file).items()
            if u in user2emb
        }
        g = nx.from_dict_of_lists(archtype2conn)
        archt_users = set(archtype2conn.keys())
        archet_ids = [i for i, u in enumerate(g.nodes()) if u in archtype2conn]
        return user2emb, g, archet_ids, archt_users

    def create_archetypes_graph_embeddings(self, graph_model: myGAT):
        """Run GAT on archetypes initial embeddings to get graph aware
        embeddigns."""
        user2emb, g, archetypes_ids, archetype_users = self.get_adjacency_and_features()
        init_embs = Tensor([user2emb[user] for user in g.nodes()])
        adj_coo = Tensor(nx.adjacency_matrix(g).todense()).to_sparse_coo().indices()
        adj_coo = to_undirected(adj_coo)
        adj_coo = add_self_loops(adj_coo)[0]
        graph = Data(x=init_embs, edge_index=adj_coo)

        graph_model.model.eval()
        with torch.no_grad():
            embs_graph = graph_model.model(graph.x, graph.edge_index.contiguous())
        archetypes_graph_embs = embs_graph[archetypes_ids].tolist()

        FileIO.write_json(
            {user: emb for user, emb in zip(archetype_users, archetypes_graph_embs)},
            self.user2graph_emb_file
        )
