from pydantic import Field
from pydantic_settings import BaseSettings


class GatConfig(BaseSettings):
    name: str = Field("samGAT", description="Name of the experiment")
    seed: int = Field(1337, description="Random seed for deterministic runs")
    in_size: int = Field(100, description="GAT model input size")
    hidden_size: int = Field(100, description="GAT hidden size")
    num_layers: int = Field(2, description="Num gat convolution layers")
    num_heads: int = Field(1, description="Num gat convolution layers")
    out_size: int = Field(100, description="GAT out size")
    unitary: bool = Field(False, description="If True, the graph embeddings will be unitary")
    dropout: float = Field(0.3, description="probability of dropout of GAT")
