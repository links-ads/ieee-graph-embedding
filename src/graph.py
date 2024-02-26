import logging
from typing import Union, List, Iterator
import torch
import random
from torch import Tensor, LongTensor
from torch_geometric.data import Data
from torch_geometric.sampler import NeighborSampler, NodeSamplerInput


logger = logging.getLogger("graphembedding")


class NegativesSampler():
    """Class to sample M nodes that are not in the neighborhood of the given anchors.
    TODO Currently implemented in a stupid but efficient way, i.e., sample random nodes 
    and hope they are not in the neigh. of the anchors (which is reasonable assumption 
    when n_ahcors is low and n_nodes is big, and average degree is low)
    """
    def __init__(self, data: Data, N: int) -> None:
        """
        :param Data data:  Graph data
        :param int N: Number of negatives to sample for each anchor id
        """
        self.N = N
        self.pad_id = -1
        self.data = data

    def sample_non_neighs_of(self, anchors: Union[LongTensor, List[int]] = None) -> None:
        n_anchors = len(anchors)
        self.sizes = [self.N for _ in range(n_anchors)]
        self.flat_ids_nopad = LongTensor(
            random.sample(range(self.data.num_nodes), self.N * n_anchors)
        )
        self.num_nodes = len(self.flat_ids_nopad)


class PositivesSampler():
    """Class to sample N nearest neighbours of given anchors"""
    def __init__(self, data: Data, N: int) -> None:
        """
        :param Data data: Graph data
        :param int N: Number of positives to sample for each anchor id
        """
        self.N = N
        self.pad_id = -1
        self.sampler = NeighborSampler(data, num_neighbors=[N])

    def sample_neghborhood_of(self, anchors: Union[LongTensor, List[int]]) -> None:
        """
        :param Union[LongTensor, List[int]] anchors: List of anchors node ids
        :param bool padding: If True, pads to N the number of positives
        :returns: List of neighborhoods os size N (padded if needed), one for each anchor
        """
        p = self.sampler.sample_from_nodes(NodeSamplerInput(input_id=None, node=anchors))
        self.sizes, flat_relative_ids = self.get_neighbourhood_sizes(p.col, p.row)
        assert len(self.sizes) == len(anchors), "Something wrong with anchors' neighbours"
        # Remove anchors (returned by default as the first 'len(anchors)' nodes in p.node).
        self.flat_ids_nopad = p.node[flat_relative_ids]
        self.num_nodes = len(self.flat_ids_nopad)

    def get_neighbourhood_sizes(self, cols: Tensor, rows: Tensor) -> Tensor:
        """
        1) Get actual size of anchors positives neighbours ignoring self-loop, whenever they
            were sampled, that point back to the anchor itself rather than to its nearest neigh.
        2) Compute actual neighborhoods: two or more anchors might have the same neighbour 
            positive node, in this case p.node cannot be used as it only holds unique nodes.
            So actual neighborhood has to be computed from the edge_index matrix of p.
        """
        sizes, neighs = [], []
        curr_neigh_size, curr_neigh = 0, []
        prev_src = None
        for src, targ in zip(cols.tolist(), rows.tolist()):
            if prev_src is None:
                prev_src = src
            if prev_src != src:  # Source node changed so the prev neighbour is finished.
                sizes.append(curr_neigh_size)
                neighs.append(curr_neigh)
                curr_neigh_size, curr_neigh = 0, []
                prev_src = src
            if targ != src:  # If target node = source node, is self-loop -> don't count.
                curr_neigh_size += 1
                curr_neigh.append(targ)
        sizes.append(curr_neigh_size)
        neighs.append(curr_neigh)
        return sizes, [n for neigh in neighs for n in neigh]


class GraphBatcher():
    def __init__(self,
                 data: Data,
                 n_positives: int,
                 n_negatives: int,
                 num_k_hops_neigs: List[int]) -> None:
        self.data = data
        self.neigs_hop_sampler = NeighborSampler(data, num_neighbors=num_k_hops_neigs)
        self.pos_sampler = PositivesSampler(data, n_positives)
        self.neg_sampler = NegativesSampler(data, n_negatives)

    def create_batches(self, n_batches: int, batch_size: int) -> Iterator[Data]:
        n_anchors = batch_size
        for _ in range(n_batches):
            # Sample anchors, positives and negatives.
            anchor_ids = LongTensor(random.sample(range(self.data.num_nodes), n_anchors))
            self.pos_sampler.sample_neghborhood_of(anchor_ids)
            self.neg_sampler.sample_non_neighs_of(anchor_ids)

            # Sample k-hops away subgraph induced by anchor + positive + negative nodes.
            batch_node_ids = torch.cat(
                (anchor_ids, self.pos_sampler.flat_ids_nopad, self.neg_sampler.flat_ids_nopad)
            )
            s = self.neigs_hop_sampler.sample_from_nodes(NodeSamplerInput(input_id=None, node=batch_node_ids))
            batch_subgraph = self.data.subgraph(s.node)

            # Make sure the original (i.e. withouth k-hop neighbours) batch_ids nodes are preserved
            # as the first nodes in subgraph.
            num_orig_nodes = len(batch_node_ids)
            assert all([all(x) for x in self.data.x[batch_node_ids] == batch_subgraph.x[:num_orig_nodes]])

            # Build anchor-positives-negatives indexes relative to the subgraph.
            anchor_ids = torch.arange(len(anchor_ids))
            offset = len(anchor_ids)
            positive_ids = torch.arange(offset, offset + self.pos_sampler.num_nodes)
            offset += len(positive_ids)
            negative_ids = torch.arange(offset, offset + self.neg_sampler.num_nodes)

            yield batch_subgraph, anchor_ids, positive_ids, negative_ids, \
                self.pos_sampler.sizes, self.neg_sampler.sizes
