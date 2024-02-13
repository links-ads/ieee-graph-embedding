from pydantic import Field
from pydantic_settings import BaseSettings
from src.config.paths import Paths


class AppraiseDataConfig(BaseSettings):
    name: str = Field('Appraise Evaluation Dataset')
    dirpath: str = Field(f'{Paths.data_dir}/AppraiseEval')
    archetypes_dir: str = Field(f'{dirpath}/archetypes')
    archets2tweets_file: str = Field(f'{archetypes_dir}/hate_archetypes_and_neighs_tweets.json')
    archets2neighs_file: str = Field(f'{archetypes_dir}/hate_archetypes_neighbours.json')
    archets2initembs_file: str = Field(f'{archetypes_dir}/hate_archetypes_and_neighs_initial_embeddings.json')
    archets2graphembs_file: str = Field(f'{archetypes_dir}/hate_archetypes_and_neighs_graph_embeddings.json')
