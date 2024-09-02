from pydantic import Field
from pydantic_settings import BaseSettings
from src.config.paths import Paths


class AppraiseDataConfig(BaseSettings):
    name: str = Field('Appraise Evaluation Dataset')
    dirpath: str = Field(f'{Paths().data_dir}/AppraiseEval')
    archetypes_dir: str = Field(f'{dirpath}/archetypes')
    archets2tweets_file: str = Field(f'{archetypes_dir}/hate_archetypes_and_neighs_tweets.json')
    archets2neighs_file: str = Field(f'{archetypes_dir}/hate_archetypes_neighbours.json')
    archets2initembs_file: str = Field(f'{archetypes_dir}/hate_archetypes_and_neighs_initial_embeddings.json')
    archets2graphembs_file: str = Field(f'{archetypes_dir}/hate_archetypes_and_neighs_graph_embeddings.json')


class DataConfig(BaseSettings):
    data_dir: str = Paths().data_dir
    convMe_dataset: str = Field(f'{data_dir}/InternetArgumentCorpus', description='Path to training data')
    HatefulUsersOnTwitter_dataset: str = Field(f'{data_dir}/HatefulUsersOnTwitter', description='Path to training data')
    TwitterNeighbours_dataset: str = Field(f'{data_dir}/TwitterNeighbours', description='Path to training data')
    TwitterUserConvsGraph_dataset: str = Field(f'{data_dir}/TwitterUserConvsGraph')
    devgraph: bool = Field(False, description="Consider a small subgraph for quick code development and debug")
