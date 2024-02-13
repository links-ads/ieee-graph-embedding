from typing import Optional
import torch
from torch import FloatTensor, Tensor, BoolTensor
import torch.nn as nn
from src.utils import batch_cos_sim


class NCELoss(nn.Module):
    """Noise Contrastive Estimation Loss, https://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf
    NOTE: NCE is has been developed for only 1 positive example, here we use n_pos positives
    for each anchor. We solve this discrepancy by taking the avg. of the dot prod.
    with all positives. THIS IS NOT NECESSARELY CORRECT."""
    def __init__(self) -> None:
        super(NCELoss, self).__init__()
        self.logsigma = nn.LogSigmoid()

    def forward(self,
                anchor: Tensor,
                positives: Tensor,
                negatives: Tensor) -> torch.float:
        """
        Parameters
        ----------
        anchor:    2D Tensor (batch, vec_size)
        positives: 3D Tensor (batch, n_pos, vec_size)
        negatives: 3D Tensor (batch, n_negs, vec_size)

        Returns
        -------
        NCE loss: Float
        """
        # Expanding anchor empty dimension 0 to match shape of positives and negatives.
        anchor = anchor.unsqueeze(0)
        pos = torch.mm(anchor, positives.transpose(0, 1)).squeeze(0)
        neg = torch.mm(anchor, negatives.transpose(0, 1)).squeeze(0)
        a = torch.sum(self.logsigma(pos))
        b = torch.sum(self.logsigma(-neg), dim=-1)
        return - (a + b)


class NormalizedNCELoss(nn.Module):
    """NCE loss where cosine similarity is used instead of dot product,
    i.e angles are alligned instead of the whole vectors.
    """
    def __init__(self) -> None:
        super(NormalizedNCELoss, self).__init__()
        self.logsigma = nn.LogSigmoid()

    def forward(self,
                anchor: Tensor,
                positives: Tensor,
                negatives: Tensor) -> torch.float:
        """
        Parameters
        ----------
        anchor:    2D Tensor (batch, vec_size)
        positives: 3D Tensor (batch, n_pos, vec_size)
        negatives: 3D Tensor (batch, n_negs, vec_size)

        Returns
        -------
        Normalized-NCE loss: Float
        """
        assert anchor.shape[0] == positives.shape[0] == negatives.shape[0]
        batch_size = anchor.shape[0]
        # Expanding anchor empty dimension 0 to match shape of positives and negatives.
        anchor = anchor.unsqueeze(1)
        pos = batch_cos_sim(anchor, positives).squeeze(1)
        neg = batch_cos_sim(anchor, negatives).squeeze(1)
        a = torch.sum(self.logsigma(pos))
        b = torch.sum(self.logsigma(-neg))
        return - (a + b) / batch_size


class NPairLoss(nn.Module):
    """(N-pair) Contrastive loss based on
    https://proceedings.neurips.cc/paper/2016/file/6b180037abbebea991d8b1232f8a8ca9-Paper.pdf.
    """
    def __init__(self) -> None:
        super(NPairLoss, self).__init__()
        self.logsigma = nn.LogSigmoid()

    def forward(self,
                anchor: Tensor,
                positives: Tensor,
                negatives: Tensor) -> torch.float:
        '''
        Pulls anchor and positive closer together and pushes anchor and negatives further apart.
        For each example in the batch, there is 1 anchor, N_p positives and N_n negatives.
        The loss formulated here optimizes the DOCT PRODUCT.
        Parameters
        ----------
        anchor:     1D tensor - anchor embedding
        positives:  2D tensor - N_p positive embeddings
        negatives : 2D tensor - N_n negative embeddings
        Returns
        -------
        Float tensor
            Sum of N-contrastive-loss.
        '''
        # Expanding anchor empty dimension 0 to match shape of positives and negatives.
        anchor = anchor.unsqueeze(0)
        pos = torch.mm(anchor, positives.transpose(0, 1)).squeeze(0)
        neg = torch.mm(anchor, negatives.transpose(0, 1)).squeeze(0)
        a = torch.sum(torch.exp(pos), dim=-1)
        b = torch.sum(torch.exp(neg), dim=-1)
        return - torch.log(a / (a + b))


class SoftNearestNeighboursLoss(nn.Module):
    """Exstension of N-Pair Loss where also M positives are provided, based on:
    https://arxiv.org/pdf/1902.01889.pdf (where distance is used instead of cos_sim)"""
    def __init__(self, trainable_temperature: bool, metric: bool):
        super(SoftNearestNeighboursLoss, self).__init__()
        # temperature T can be either taken as static or trained with the other parameters of the model.
        if trainable_temperature:
            self.T = nn.Parameter(Tensor([1]))
        else:
            self.T = Tensor([1])
        self.metric = metric

    def _assert_batch(self, anchor: Tensor, positives: Tensor, negatives: Tensor) -> int:
        # Assert for each anchor there are Np positives and Nn negatives.
        assert len(anchor.shape) + 1 == len(positives.shape) == len(negatives.shape)
        # If single example is given (instead of batch), expand empty dimension 0.
        if len(anchor.shape) == 1:
            anchor = anchor.unsqueeze(0)
            positives = positives.unsqueeze(0)
            negatives = negatives.unsqueeze(0)
        # Assert batch dimension is the same.
        assert anchor.shape[0] == positives.shape[0] == negatives.shape[0]
        batch_size = anchor.shape[0]
        return batch_size

    def forward(self,
                anchor: FloatTensor,
                positives: FloatTensor,
                negatives: FloatTensor,
                mask_pos: Optional[BoolTensor] = None,
                mask_neg: Optional[BoolTensor] = None) -> torch.float:
        """
        Forward pass of the loss.
        :parameter mask_pos: BoolTensor - mask for excluding padding of positives
        :parameter mask_neg: BoolTensor - mask for excluding padding of negatives
        """
        batch_size = self._assert_batch(anchor, positives, negatives)
        # In masks are not provided, set them to ones, so they have no efffect.
        if mask_pos is None:
            mask_pos = torch.ones(positives.shape[0], positives.shape[1])
        if mask_neg is None:
            mask_neg = torch.ones(negatives.shape[0], negatives.shape[1])
        anchor = anchor.unsqueeze(1)
        if self.metric == 'distance':
            pos = (anchor.unsqueeze(1) - positives).norm(dim=-1) / self.T
            neg = (anchor.unsqueeze(1) - negatives).norm(dim=-1) / self.T
        elif self.metric == 'cosine':
            pos = batch_cos_sim(anchor, positives).squeeze(1) / self.T
            neg = batch_cos_sim(anchor, negatives).squeeze(1) / self.T
        elif self.metric == 'dot':
            pos = torch.bmm(anchor, positives.transpose(2, 1)).squeeze(1) / self.T
            neg = torch.bmm(anchor, negatives.transpose(2, 1)).squeeze(1) / self.T
        pos = torch.sum(torch.exp(pos) * mask_pos, dim=-1) / mask_pos.sum(-1)  # Masked average.
        neg = torch.sum(torch.exp(neg) * mask_neg, dim=-1) / mask_neg.sum(-1)
        return -torch.sum(torch.log(pos / (pos + neg))) / batch_size
