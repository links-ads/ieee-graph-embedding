import json
import torch
from pydantic import Field
from pydantic_settings import BaseSettings
from src.config.paths import Paths
from src.config.data import DataConfig
from src.config.models import GatConfig
from src.config.training import (ContrastiveTrainingConfig, LogRegTrainingConfig,
                                 OptimizerGatConfig, OptimizerLogRegConfig,
                                 SoftNNLossConfig, LogRegConfig)


class Config(BaseSettings):
    seed: int = Field(1337, description="Random seed for deterministic runs")
    name: str = Field("samGAT", description="Name of the experiment")
    gpu_no: int = Field(0, description="Number of GPU device to user.")
    device: str = Field("cuda" if torch.cuda.is_available() else "cpu", description="Device to use for training")

    paths: Paths = Paths()
    gat: GatConfig = GatConfig()
    data: DataConfig = DataConfig()
    logreg_model: LogRegConfig = LogRegConfig()
    contrastive_train: ContrastiveTrainingConfig = ContrastiveTrainingConfig()
    logreg_train: LogRegTrainingConfig = LogRegTrainingConfig()
    loss_gat: SoftNNLossConfig = SoftNNLossConfig()
    optimizer_gat: OptimizerGatConfig = OptimizerGatConfig()
    optimizer_logreg: OptimizerLogRegConfig = OptimizerLogRegConfig()

    def prettyprint(self):
        print('\n\t----------------------------------    CONFIGS    -----------------------------')
        print(json.dumps(self.model_dump(), sort_keys=True, indent=4))
        print('\t------------------------------------------------------------------------------\n\n')

    def set_device_num(self):
        if self.device == 'cuda':
            self.device = f"cuda:{self.gpu_no}"
