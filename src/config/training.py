from typing import List
from pydantic import Field, BaseModel


class OptimizerGatConfig(BaseModel):
    name: str = Field('Adam', description='Name of the optimizer used for training.')
    lr: float = Field(1e-3, description="Learning rate")
    momentum: float = Field(0.9, description="Momentum for SGD")
    weight_decay: float = Field(1e-2, description="Weight decay for the optimizer")


class OptimizerLogRegConfig(BaseModel):
    name: str = Field('Adam', description='Name of the optimizer used for training.')
    lr: float = Field(1e-1, description="Learning rate")
    momentum: float = Field(0.9, description="Momentum for SGD")
    weight_decay: float = Field(1e-3, description="Weight decay for the optimizer")


class LogRegConfig(BaseModel):
    model: str = Field("LogReg", description='Model identifier')
    input_size: int = Field(10, description='Input size of the linear classifier, has to match the output embedding size of GAT model')


class ContrastiveTrainingConfig(BaseModel):
    n_epochs: int = Field(100, description='Max number of training epochs')
    n_batches: int = Field(64, description='Number of batches considered each training epoch')
    batch_size: int = Field(32, description='Number of nodes ANCHOR nodes per batch. NOTE: each anchor node requires N positive and M negatives, each of those require L nearest neighbours for the model to run')
    patience: int = Field(5, description="Number of epochs without improvement before early stopping")
    learnable_temperature: bool = Field(False, description="If True, the temperature of SoftNN loss is trained together with the model")
    n_positives: int = Field(20, description="Number of positive examples for each anchor")
    n_negatives: int = Field(20, description="Number of negative examples for each anchor")
    num_k_hops_neighs: List[int] = Field([10, 5, 5], description="Number of neighbours to sample for each hop")
    metric: str = Field('cosine', description="What metric to use in the contrastive loss")


class AutoencoderTrainingConfig(BaseModel):
    n_epochs: int = Field(100, description='Max number of training epochs')


class LogRegTrainingConfig(BaseModel):
    n_epochs: int = Field(100, description='Max number of training epochs')
    batch_size: int = Field(16, description='Number of nodes ANCHOR nodes per batch. NOTE: each anchor node requires N positive and M negatives, each of those require L nearest neighbours for the model to run')
    patience: int = Field(5, description="Number of epochs without improvement before early stopping")


class SoftNNLossConfig(BaseModel):
    name: str = Field('SoftNearestNeigbourLoss', desctiption='Soft NN loss for contrastive learning.')
    trainable_temperature: bool = Field(True, description='Train temperature of SoftNN contrastive loss.')


class BCELossConfig(BaseModel):
    name: str = Field('BinaryCrossEntropyLoss', desctiption='Binary Cross Entropy loss for supervised learning with 2 classes.')
