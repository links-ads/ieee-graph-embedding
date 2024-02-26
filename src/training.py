import logging
from src.utils import seed_everything, set_savefolder, FileIO, split_graph_train_val
from src.datasets import TwitterNeighbours
from src.models import samGAT
from src.modelling import ContrastiveGraphTrainer
from src.config.main import Config

logger = logging.getLogger('graphembedding')


def train_gat_contrastive(config: Config) -> None:
    config.set_device_num()
    config.prettyprint()

    seed_everything(config.gat.seed)
    savedir = set_savefolder(config.paths.output_folder,
                             config.gat.name,
                             str(config.gat_standard_model))
    FileIO.write_json(config.model_dump(), f'{savedir}/configs.json')  # Save configs.
    # tb_logger = SummaryWriter(log_dir=f'{savedir}/tb-logs', comment="000")  # Tb loagger.
    # res_logger = ResultsLogger(savedir, config.timestamp)

    # Load data and Remove self-loops and isolated nodes.
    dataset = TwitterNeighbours(
        folder=config.data.HatefulUsersOnTwitter_dataset,
        undirected=True,
        remove_selfloops=True,
        rm_isolated_nodes=True
    )
    graph_train, graph_val = \
        split_graph_train_val(dataset.data, train_ratio=0.7)

    gat = samGAT(
        in_size=dataset.data.num_node_features,
        emb_size=config.gat.hidden_size,
        n_heads=config.gat.num_layers,
        out_size=config.gat.out_size,
        num_layers=config.gat.num_layers,
        dropout=config.gat.dropout,
        unitary=config.gat.unitary
    ).to(config.device)

    trainer = ContrastiveGraphTrainer(
        graph_eval=graph_val,
        folder=dataset.folder,
        device=config.device
    )

    gat = trainer.train(
        graph=graph_train,
        encoder=gat,
        loss_n_pos=6,
        loss_n_negs=30,
        n_epochs=config.contrastive_train.n_epochs,
        lr=config.optimizer_gat.lr,
    )

    gat.save(savedir)
