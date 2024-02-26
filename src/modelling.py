import time
import logging
from typing import Tuple, List, Iterable, Optional, Union
from copy import deepcopy
from sklearn.metrics import precision_recall_fscore_support as prec_rec_f_supp
from random import shuffle
import torch
import torch.nn as nn
from torch.optim import Adam
from torch import Tensor, LongTensor
from torch import cosine_similarity as cos_sim
from torch_geometric.data import Data
from src.models.gat.models import samGAT
from src.graph import GraphBatcher, PositivesSampler
from src.models.decoders import LogisticRegression
from src.loss import SoftNearestNeighboursLoss
from src.results import ResultsLogger
from src.utils import get_neighbours, compute_furthest_nodes, FileIO
import warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn


logger = logging.getLogger('graphembedding')


class Batcher:
    @staticmethod
    def create_batches(data: Tuple[Tensor, LongTensor],
                       size: int,
                       shuffle: bool = True) -> Iterable[Tuple[Tensor, LongTensor]]:
        inputs, targets = data
        n = len(inputs)

        if shuffle:
            shufl_idxs = torch.randperm(n)
            inputs, targets = inputs[shufl_idxs], targets[shufl_idxs]

        n_batches = n // size + int(n % size != 0)
        for i in range(n_batches):
            yield inputs[i * size: (i + 1) * size], targets[i * size: (i + 1) * size]


class SupervisedTrainer:
    def __init__(self, patience: int, device: str):
        super(SupervisedTrainer, self).__init__()
        self.device = device
        self.patience = patience
        self.patience_count = 0
        self.delta = 0
        self.criterion = nn.BCEWithLogitsLoss()
        self.best_metric = 0
        self.best_model = None
        self.train_msg = "Train L: %.3f | Valid  L: %.3f | Prec: %.3f | Rec: %.3f " + \
            "| F1: %.3f | Pos: %.3f%%"

    def train(self, model: LogisticRegression,
              data_train: Tuple[Tensor, LongTensor],
              data_valid: Tuple[Tensor, LongTensor],
              lr: float,
              n_epochs: int,
              batch_size: float):
        """Standard supervised training with BCE loss.
        Stop after validation loss hasn't improved for 'patience' epochs."""
        self.best_model = model
        optimizer = Adam(model.parameters(), lr=lr)

        # Training.
        for epoch in range(1, n_epochs + 1):
            model.train()

            for inps_b, targs_b in Batcher.create_batches(data_train, size=batch_size):
                optimizer.zero_grad()
                preds = model(inps_b.to(self.device))
                loss_train = self.criterion(preds, targs_b.type(torch.float32).to(self.device))
                loss_train.backward()
                optimizer.step()

            loss_eval, p, r, f1, supp = self.evaluation(model, data_valid)
            # logger.debug(self.train_msg % (loss_train.item(), loss_eval, p, r, f1, supp))

            if not self.has_improved(f1, model):
                break

        # logger.debug(f'LogReg training finished! Best F1: {self.best_metric}')
        return self.best_model

    def evaluation(self, model: LogisticRegression, data_valid) -> Tuple:
        inps, targets = data_valid
        model.eval()
        with torch.no_grad():
            logits = model(inps.to(self.device))
        loss = self.criterion(logits, targets.type(torch.float32).to(self.device)).item()
        preds = model.sigma(logits).round()
        p, r, f1, _ = prec_rec_f_supp(targets.cpu(), preds.cpu(), average='binary', pos_label=1)
        perc_ones = targets.sum() / len(targets)
        return loss, p, r, f1, perc_ones

    def has_improved(self, metric: float, model: LogisticRegression):
        """Check if the 'Metric Of Reference' has improved in the last 'patience' steps,
        and save best model."""
        if metric == 0:
            return True

        if metric - self.best_metric > self.delta:  # Loss improved (more than delta).
            self.best_metric = metric
            self.best_model = deepcopy(model)
            self.patience_count = 0
            return True

        elif self.patience_count < self.patience:  # Loss didn't improve but still patience.
            self.patience_count += 1
            return True

        return False


def pad_neighbourhood():
    pass


class GraphTrainer():
    def __init__(self, data: Data, device: str) -> None:
        self.data = data
        self.device = device

    def train_logistic(self,
                       embeddings: Tensor,
                       labels: LongTensor,
                       patience: int,
                       lr: float,
                       batch_size: int,
                       n_epochs: int):
        """Supervised training of the Logitic Regression model.
        Given (frozen) node embeddings as input, predict their label
        training runs on the subset of labelled data only

        :param 2D FloatTensor embeddings: Frozen node embeddings
        :param 1D LongTensor labels: Labels of embeddings
        :param float lr: Learning rate for logistic regression
        :param int batch_size: Batch_size of logistic regression
        :param int n_epochs: Number of epochs for training logreg
        :param str device: Either cpu or cuda
        :return float p: Precision of logreg on test set
        :return float r: Recall of logreg on test set
        :return float f1: F1 score of logreg on test set
        """
        # logger.debug(' ---------- LogReg ----------')
        n_labels = labels.unique().shape[0]
        assert n_labels == 2, f'2 unique label values expected, got: {n_labels}'
        assert labels.shape[0] == embeddings.shape[0], 'Mismatch in num. of labels and embeds'

        # Split shuffled data into train/val/test set.
        n = embeddings.shape[0]
        tr, val, ts = int(0.6 * n), int(0.15 * n), int(0.25 * n)
        train_data = (embeddings[: tr], labels[: tr])
        valid_data = (embeddings[tr: tr + val], labels[tr: tr + val])
        test_data = (embeddings[-ts:], labels[-ts:])

        # Train.
        trainer = SupervisedTrainer(patience, self.device)
        model = LogisticRegression(embeddings.shape[-1], self.device)
        model = trainer.train(model, train_data, valid_data, lr, n_epochs, batch_size)

        # Test.
        embs, targets = test_data
        preds = model.predict(embs.to(self.device))
        n_positives = targets.sum() / len(targets)
        p, r, f1, _ = \
            prec_rec_f_supp(targets.cpu(), preds.cpu(), average='binary', pos_label=1)
        logger.debug('LogReg Test - Prec: %.3f, Recall: %.3f, F1: %.3f, Pos %.3f%%' % (p, r, f1, n_positives))
        return p, r, f1

    def train_contrastive(self,
                          model: samGAT,
                          n_positives: int,
                          n_negatives: int,
                          num_k_hops_neighs: List[int],
                          trainable_temperature: bool,
                          lr: float,
                          weight_decay: float,
                          metric: str,
                          n_epochs: int,
                          n_batches: int,
                          batch_size: int,
                          patience: int,
                          lr_logreg: float,
                          patience_logreg: int,
                          bs_logreg: int,
                          nepochs_logreg: int,
                          res_logger: Optional[ResultsLogger]) -> samGAT:
        # Set number of k-hops for based on number of GatConv layers.
        assert model.num_layers <= len(num_k_hops_neighs), "More k-hops-neighbours are \
            required to be able to apply all GatConv layers"
        num_k_hops_neighs = num_k_hops_neighs[: model.num_layers]  # Only get as many k-hops neighs as needed for GatConv.

        # Set loss and optimizer.
        # pad_emb = torch.zeros(model.emb_size, requires_grad=False).to(self.device)
        pad_emb = torch.zeros(100, requires_grad=False).to(self.device)

        criterion = SoftNearestNeighboursLoss(trainable_temperature, metric).to(self.device)
        optimizer = Adam([
            {'params': model.parameters(), "lr": lr, "weight_decay": weight_decay},
            {'params': criterion.parameters(), "lr": lr}
        ])

        # Mini-batch training.
        gbatcher = GraphBatcher(self.data, n_positives, n_negatives, num_k_hops_neighs)
        best_f1, patience_count = 0, 0
        for epoch in range(n_epochs):
            logger.debug(f' -------- epoch {epoch} -------- ')
            start = time.time()
            model.train()
            loss_epoch, T = 0, 0
            n_batches = n_batches

            for gbatch in gbatcher.create_batches(n_batches, batch_size):
                optimizer.zero_grad()

                subgraph, anchor_ids, positive_ids, negative_ids, pos_sizes, neg_sizes = gbatch
                subgraph = subgraph.to(self.device)
                graph_embs = model(subgraph.x, subgraph.edge_index.contiguous())
                # Get anchors, padded positives and padded negatives.
                anchors = graph_embs[anchor_ids]
                positives, pos_pad_mask = pad_neighbourhood(
                    graph_embs[positive_ids], n_positives, pos_sizes, pad_emb)
                negatives, neg_pad_mask = pad_neighbourhood(
                    graph_embs[negative_ids], n_negatives, neg_sizes, pad_emb)
                # Loss.
                loss = criterion(anchors, positives, negatives, pos_pad_mask, neg_pad_mask)
                loss_epoch += loss.item()
                T += criterion.T.item()

                if epoch > 0:  # Epoch 0 DO NOT TRAIN the model!
                    loss.backward()
                    optimizer.step()
                    # logger.debug(f'epoch: {epoch},  loss: {loss.item()}')
            logger.debug('Contrastive Loss train: %.3f,    T: %.3f' %
                         (loss_epoch / n_batches, T / n_batches))

            SUPERVISED_EVAL = False
            if SUPERVISED_EVAL:
                p, r, f1 = self.evaluation_supervised(model, patience_logreg, lr_logreg,
                                                      bs_logreg, nepochs_logreg)
                # Log results.
                if res_logger is not None:
                    res_logger.write_epoch_preformance(epoch, p, r, f1)
            else:
                align_pos, align_neg = self.evaluation_unsupervised(model, criterion)
                f1 = 2 * (align_pos - align_neg) / (align_pos + align_neg)

            logger.debug('time: %d sec.   Alignment pos: %3f, align neg: %3f, f1: %3f' % (
                (time.time() - start), align_pos, align_neg, f1)
            )

            # Check model improvement with patience.
            if f1 > best_f1:
                logger.debug(f'Best Epoch - alignment: {f1} !')
                patience_count = 0
                best_f1 = f1
            if f1 <= best_f1:
                if patience_count < patience:
                    patience_count += 1
                else:
                    logger.debug('Early stopping!')
                    break

        return model

    def evaluation_supervised(self, model: samGAT, patience_logreg: int,
                              lr_logreg: float, bs_logreg: int, nepochs_logreg: int):
        model.eval()
        # Get evaluation subgraph.
        eval_ids = (self.data.y != 2).nonzero(as_tuple=True)[0]  # 2 means unlabelled in the HUT dataset.
        eval_subgraph = self.data.subgraph(eval_ids).to(self.device)
        with torch.no_grad():
            eval_embs = model(eval_subgraph.x, eval_subgraph.edge_index.contiguous())

        return self.train_logistic(eval_embs, eval_subgraph.y, patience_logreg, lr_logreg,
                                   bs_logreg, nepochs_logreg)

    def evaluation_unsupervised(self, model: samGAT, criterion):
        "Measures allignment with positives and negatives on test set."
        def _get_negatives(all_nodes, anchor, positives):
            """Get mask for negative indices as every node that is not anchor or positives"""
            anchor = LongTensor([anchor])
            negative_ids = torch.hstack([anchor, positives])
            mask = torch.ones(all_nodes.numel(), dtype=torch.bool)
            mask[negative_ids] = False
            return all_nodes[mask]

        model.eval()
        eval_ids = (self.data.y == 1).nonzero(as_tuple=True)[0]  # 1 means test set in the TwitterUsers dataset.
        eval_subgraph = self.data.subgraph(eval_ids).to(self.device)

        sampler = PositivesSampler(eval_subgraph, N=1000)
        sampler.sample_neghborhood_of(torch.LongTensor(range(eval_subgraph.num_nodes)))
        all_nodes = sampler.flat_ids_nopad.clone()

        with torch.no_grad():
            graph_embs = model(eval_subgraph.x, eval_subgraph.edge_index.contiguous())

        alignment_pos, alignment_neg = 0, 0
        offset = 0
        for anchor, size in enumerate(sampler.sizes):
            positives = sampler.flat_ids_nopad[offset: offset + size]
            negatives = _get_negatives(all_nodes, anchor, positives)
            anchor_emb = graph_embs[anchor]
            positive_embs = graph_embs[positives]
            negative_embs = graph_embs[negatives]

            alignment_pos += torch.mm(anchor_emb.unsqueeze(0), positive_embs.T).mean().item()
            alignment_neg += torch.mm(anchor_emb.unsqueeze(0), negative_embs.T).mean().item()

            # alignment_pos += cos_sim(anchor_emb, positive_embs).mean().item()
            # alignment_neg += cos_sim(anchor_emb, negative_embs).mean().item()

        return alignment_pos / len(sampler.sizes), alignment_neg / len(sampler.sizes)


class ContrastiveGraphTrainer:
    def __init__(self, graph_eval: Data, folder: str, device: str) -> None:
        self.graph_eval = graph_eval
        self.device = device
        self.evaluator = Evaluator(graph_eval, folder, False, False)
        self.criterion = SoftNearestNeighboursLoss(False, 'dot').to(self.device)
        self.criterion.T = 0.3
        self.neighbour_nodes = \
            FileIO.read_json(f'{folder}/useruser_neighbour_nodes.json')

    def train(self,
              graph: Data,
              encoder: samGAT,
              loss_n_pos: int,
              loss_n_negs: int,
              n_epochs: int,
              lr: float) -> samGAT:
        # Initialize the optimizer.
        optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)

        # Train the model.
        for epoch in range(n_epochs):
            logger.info(f'Epoch {epoch}')
            optimizer.zero_grad()
            embs = encoder(x=graph.init_emb,
                           edge_index=graph.edge_index.contiguous())
            loss = self.compute_contrastive_loss(embs, loss_n_pos, loss_n_negs)
            logger.info('Loss contrastive train: %.5f' % loss.item())
            self.evaluator.evaluate(encoder)
            loss.backward()
            optimizer.step()

        return encoder

    def compute_contrastive_loss(self, embs: Tensor, n_pos: int,
                                 n_negs: int) -> Tensor:
        pad_emb = torch.zeros(embs.shape[1])
        nodes_ids = list(range(embs.shape[0]))
        positives, negatives, pos_pad_mask = [], [], []
        for i in nodes_ids:
            # Sample n_pos positives.
            neigh_ids = self.neighbour_nodes[str(i)]
            shuffle(neigh_ids)
            neigh_ids = neigh_ids[: n_pos]
            # Pad positives.
            n_pad = max(0, n_pos - len(neigh_ids))
            pos_pad_mask.append([1] * len(neigh_ids) + [0] * n_pad)
            # Sample n_negs negatives.
            non_neighs = list(set(nodes_ids) - set(neigh_ids))
            shuffle(non_neighs)
            # Get pos and neg embs.
            positives.append(
                torch.vstack(
                    [embs[neigh_ids[: n_pos]], pad_emb.repeat(n_pad, 1)]
                )
            )
            negatives.append(embs[non_neighs[: n_negs]])
        anchors = embs
        positives = torch.stack(positives)
        negatives = torch.stack(negatives)
        pos_pad_mask = torch.BoolTensor(pos_pad_mask)

        return self.criterion(anchors, positives, negatives, pos_pad_mask)


class Evaluator:
    def __init__(self,
                 graph_eval: Data,
                 folder: str,
                 compute_furthest_nodes_dict: bool = False,
                 compute_neigbours_dict: bool = False) -> None:
        self.graph_eval = graph_eval
        logger.info('Initializing Evaluator...')
        if compute_furthest_nodes_dict:
            logger.info('Computing furthest nodes...')
            self.furthest_nodes = compute_furthest_nodes(graph_eval)
            FileIO.write_json(self.furthest_nodes,
                              f'{folder}/useruser_furthest_nodes.json')
        else:
            self.furthest_nodes = \
                FileIO.read_json(f'{folder}/useruser_furthest_nodes.json')

        if compute_neigbours_dict:
            logger.info('Computing neighbours...')
            self.neighbour_nodes = {
                node_idx: get_neighbours(graph_eval, node_idx).tolist()
                for node_idx in range(graph_eval.num_nodes)
            }
            FileIO.write_json(self.neighbour_nodes,
                              f'{folder}/useruser_neighbour_nodes.json')
        else:
            self.neighbour_nodes = \
                FileIO.read_json(f'{folder}/useruser_neighbour_nodes.json')

    def evaluate(self, encoder: samGAT, precomputed_embs: Tensor = None
                 ) -> Tuple[float, float]:
        if precomputed_embs is None:
            with torch.no_grad():
                embs_eval = encoder(self.graph_eval.init_emb,
                                    self.graph_eval.edge_index.contiguous())
        else:
            embs_eval = precomputed_embs
        """Evaluate the model on the graph."""
        neigh_simil_init, neigh_simil_gat = self.similarity_users_neighbours(
            self.graph_eval, embs_eval, self.graph_eval.init_emb
        )
        far_simil_init, far_simil_gat = self.similarity_users_far_away(
            self.graph_eval, embs_eval, self.graph_eval.init_emb
        )

        # Log results.
        logger.info('simil neighs init: %.3f,  simil neighs gat: %.3f' % (
            neigh_simil_init, neigh_simil_gat)
        )
        logger.info('simil far init: %.3f,  simil far gat: %.3f' % (
            far_simil_init, far_simil_gat)
        )
        return neigh_simil_init, neigh_simil_gat, far_simil_init, far_simil_gat

    def similarity_users_neighbours(self,
                                    graph_uu: Data,
                                    embs_gat: Tensor,
                                    embs_init: Tensor) -> Tuple[float, float]:
        """Compute cosine similarity between initial user embeddigs
        and graph user emgeddings of each user and its neigbours. """
        N = graph_uu.num_nodes
        cosim_init = 0
        cossim_gat = 0
        for node_idx in range(graph_uu.num_nodes):
            neighs_idx = self.neighbour_nodes[str(node_idx)]
            node_init_emb = embs_init[node_idx]
            neighs_init_emb = embs_init[neighs_idx]

            node_gat_emb = embs_gat[node_idx]
            neighs_gat_emb = embs_gat[neighs_idx]

            cosim_init += cos_sim(node_init_emb, neighs_init_emb, dim=-1).mean()
            cossim_gat += cos_sim(node_gat_emb, neighs_gat_emb, dim=-1).mean()

        return cosim_init / N, cossim_gat / N

    def similarity_users_far_away(self,
                                  graph_uu: Data,
                                  embs_gat: Tensor,
                                  embs_init: Tensor) -> Tuple[float, float]:
        """Compute cosine similarity between each User and the User that is the
        furthest away from it."""
        N = graph_uu.num_nodes
        cosim_init = 0
        cossim_gat = 0
        for node_idx in range(graph_uu.num_nodes):
            furthest_nodes_idx = self.furthest_nodes[str(node_idx)]
            node_init_emb = embs_init[node_idx]
            furthest_nodes_init_emb = embs_init[furthest_nodes_idx]

            node_gat_emb = embs_gat[node_idx]
            furthest_nodes_gat_emb = embs_gat[furthest_nodes_idx]

            cosim_init += cos_sim(node_init_emb, furthest_nodes_init_emb, dim=-1).mean()
            cossim_gat += cos_sim(node_gat_emb, furthest_nodes_gat_emb, dim=-1).mean()

        return cosim_init / N, cossim_gat / N
