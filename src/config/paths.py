from os.path import dirname, abspath
from pydantic import Field
from pydantic_settings import BaseSettings


class Paths(BaseSettings):
    config_dir: str = dirname(abspath(__file__))
    src_dir: str = dirname(config_dir)
    main_dir: str = dirname(src_dir)
    appraise_dir: str = dirname(main_dir)
    data_dir: str = Field(f'{appraise_dir}/graphembeddings/datasets')
    output_folder: str = Field(f"{main_dir}/outputs")
    model_saves_folder: str = Field(f"{main_dir}/model_saves")
    enet_folder: str = Field(f"{main_dir}/data_enetwork")
