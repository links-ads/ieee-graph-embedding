from os.path import dirname, abspath


class Paths:
    config_dir: str = dirname(abspath(__file__))
    src_dir: str = dirname(config_dir)
    main_dir: str = dirname(src_dir)
    data_dir: str = f"{main_dir}/data"
    output_folder: str = f"{main_dir}/outputs"
    model_saves_folder: str = f"{main_dir}/model_saves"
    enet_folder: str = f"{main_dir}/data_enetwork"
