import logging
import torch
from src.utils import seed_everything, set_savefolder, FileIO, split_graph_nodes_train_val
from src.datasets import HateOnTwitter
from src.models import samGAT
from src.modelling import ContrastiveGraphTrainer
from src.config.main import Config

logger = logging.getLogger('graphembedding')


def train_gat_contrastive(config: Config, load_saved_graphs: bool) -> None:
    config.set_device_num()
    config.prettyprint()

    seed_everything(config.gat.seed)
    savedir = set_savefolder(config.paths.output_folder,
                             config.gat.name,
                             'samgat-contrastive')
    FileIO.write_json(config.model_dump(), f'{savedir}/configs.json')  # Save configs.
    # tb_logger = SummaryWriter(log_dir=f'{savedir}/tb-logs', comment="000")  # Tb loagger.
    # res_logger = ResultsLogger(savedir, config.timestamp)

    # Load data and Remove self-loops and isolated nodes.
    dataset = HateOnTwitter(
        folder=config.data.HatefulUsersOnTwitter_dataset,
        load_from_saved=True,
        undirected=True,
        remove_selfloops=True,
        rm_isolated_nodes=True
    )

    if load_saved_graphs:
        graph_train = torch.load(f'{dataset.folder}/graph_train.pygeo')
        graph_val = torch.load(f'{dataset.folder}/graph_val.pygeo')
    else:
        graph_train, graph_val = \
            split_graph_nodes_train_val(dataset.data, train_ratio=0.7)
        # Save the graph.
        torch.save(graph_train, f'{dataset.folder}/graph_train.pygeo')
        torch.save(graph_val, f'{dataset.folder}/graph_val.pygeo')

    gat = samGAT(
        in_size=dataset.data.num_node_features,
        emb_size=config.gat.hidden_size,
        num_layers=config.gat.num_layers,
        n_heads=config.gat.num_heads,
        out_size=config.gat.out_size,
        dropout=config.gat.dropout,
        unitary=config.gat.unitary
    ).to(config.device)

    trainer = ContrastiveGraphTrainer(
        graph_train=graph_train,
        graph_eval=graph_val,
        folder=dataset.folder,
        device=config.device,
        compute_neighs_and_furthest_nodes_train=True,
        compute_neighs_and_furthest_nodes_eval=False
    )

    gat = trainer.train(
        encoder=gat,
        n_pos=6,
        n_negs=30,
        n_epochs=config.contrastive_train.n_epochs,
        lr=config.optimizer_gat.lr,
    )

    # gat.save(savedir)
