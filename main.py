import logging

from clidantic import Parser

import src.testing as testing
from src.training import train_gat_contrastive
from src.config.main import Config

logger = logging.getLogger('victims')

cli = Parser()


@cli.command()
def train(config: Config):
    train_gat_contrastive(config)

@cli.command()
def test():
    testing.test()


if __name__ == '__main__':
    cli()