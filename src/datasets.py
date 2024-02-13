import logging
import torch
import networkx as nx
import numpy as np
import pandas as pd
from collections import defaultdict
from src.utils import FileIO
from torch_geometric.data import Data
from torch_geometric.utils import (to_undirected, contains_isolated_nodes,
                                   remove_self_loops, remove_isolated_nodes)

logger = logging.getLogger('graphembedding')


class HateOnTwitter():
    """Dataset loading HUT data that have been already preprocessed by HateOnTwitter_old"""
    def __init__(self, folder: str,
                 load_from_saved: bool,
                 undirected: bool,
                 remove_selfloops: bool,
                 rm_isolated_nodes) -> None:
        super(HateOnTwitter, self).__init__()
        self.folder = folder
        # Load data.
        if not load_from_saved:
            self.import_from_networkx()
        self.data = torch.load(f'{self.folder}/user_clean_graph.pygeodata')
        # Preprocessing.
        if undirected:
            self.data.edge_index = to_undirected(self.data.edge_index)
        if remove_selfloops:
            self.data.edge_index, _ = remove_self_loops(edge_index=self.data.edge_index)
        if rm_isolated_nodes and contains_isolated_nodes(self.data.edge_index):
            edge_index, _, node_mask = remove_isolated_nodes(self.data.edge_index)
            self.data = Data(
                self.data.x[node_mask], edge_index=edge_index, y=self.data.y[node_mask]
            )

    def load_processed_graph_files(self):
        ids = [x.strip() for x in FileIO.read_text(f'{self.folder}/user2id.txt')]
        embs = torch.load(f'{self.folder}/embeddings.pt')
        labels = torch.load(f'{self.folder}/labels.pt')
        ids_ann = [x.strip() for x in FileIO.read_text(f'{self.folder}/user_annotated2id.txt')]
        embs_annotated = torch.load(f'{self.folder}/embs_annotated.pt')
        labels_annotated = torch.load(f'{self.folder}/labels_annotated.pt')
        g = nx.read_graphml(f'{self.folder}/users_clean.graphml')
        return g, embs, labels

    def import_from_networkx(self) -> None:
        logger.info("Loading graph from networkx")
        g, embs, labels = self.load_processed_graph_files()
        adj = nx.to_scipy_sparse_matrix(g, format='coo')
        adj_coo = torch.LongTensor(np.vstack((adj.row, adj.col)))
        data = Data(x=embs, edge_index=adj_coo, y=labels)
        torch.save(data, f'{self.folder}/user_clean_graph.pygeodata')


class TwitterNeighbours:
    def __init__(self,
                 folder: str,
                 undirected: bool,
                 remove_selfloops: bool,
                 rm_isolated_nodes) -> None:
        self.folder = folder
        self.data = torch.load(f'{self.folder}/graph_train_and_test.pygeodata')

        # Preprocessing.
        if undirected:
            self.data.edge_index = to_undirected(self.data.edge_index)
        if remove_selfloops:
            self.data.edge_index, _ = remove_self_loops(edge_index=self.data.edge_index)
        if rm_isolated_nodes and contains_isolated_nodes(self.data.edge_index):
            edge_index, _, node_mask = remove_isolated_nodes(self.data.edge_index)
            self.data = Data(self.data.x[node_mask], edge_index=edge_index)


class ConvinceMe:
    def __init__(self, folder: str) -> None:
        self.folder = folder
        self.filename = f'{self.folder}/convinceme_graph_data_dump.csv'

    def load_data_into_graph(self):
        user2conv = defaultdict(set)
        user2post = defaultdict(set)
        conv2post = defaultdict(set)
        post2text = {}
        data = pd.read_csv(self.filename).to_dict('records')
        for row in data:
            user2post[row['author_id']].add(row['text_id'])
            conv2post[row['discussion_id']].add(row['text_id'])
            user2conv[row['author_id']].add(row['discussion_id'])
            post2text[row['text_id']] = row['text']
