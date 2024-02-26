import os
from glob import glob
from typing import Union, Tuple
from datetime import datetime
import pandas as pd
from src.utils import FileIO, extract_nums_from_string
from src.config.main import Config


class ResultsLogger:
    def __init__(self, foldername: str, timestamp: int) -> None:
        self.folder = ResultsLogger.get_folder(foldername)
        self.results_file = f'{self.folder}/results.txt'
        run_time = str(datetime.fromtimestamp(timestamp))
        FileIO.write_text(run_time, self.results_file)

    def write_epoch_preformance(self, epoch: int, prec: float, rec: float, f1: float):
        FileIO.append_text(
            f'Epoch: {epoch} -- prec: {prec}, rec: {rec}, f1: {f1}',
            self.results_file
        )

    def write_baseline_performance(self, prec: float, rec: float, f1: float):
        FileIO.append_text(
            f'Baseline (no graph - only GloVe embs) -- prec: {prec}, rec: {rec}, f1: {f1}',
            self.results_file
        )

    @staticmethod
    def get_folder(foldername: str):
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        return foldername


class ResultsParser:
    def __init__(self, config: Config) -> None:
        self.results_dir = config.saves.results_dir
        self.results_df = pd.DataFrame(columns=[
            'hidden-size', 'n-heads', 'trained-T', 'lr', 'batch-size', 'n-batches',
            'base-prec', 'base-rec', 'base-f1', 'epoch', 'prec', 'rec', 'f1',
            'p-improve', 'r-improve', 'f1-improve'
        ])

    def parse_results(self) -> None:
        for foldername in glob(f'{self.results_dir}/*'):
            if not os.path.isdir(foldername):
                continue
            h, heads, lr, b_size, n_batch = extract_nums_from_string(foldername)
            T = 'False' not in foldername  # 'False' can appear only as attribute of trainedT in the filename.
            base_p, base_r, base_f1, epoch, p, r, f1 = self.parse_result_file(foldername)
            if base_f1 == 0:
                continue
            p_improve, r_improve, f1_improve = (p - base_p) / base_p, (r - base_r) / base_r, (f1 - base_f1) / base_f1
            self.results_df.loc[len(self.results_df)] = [
                h, heads, T, lr, b_size, n_batch, base_p, base_r, base_f1,
                epoch, p, r, f1, p_improve, r_improve, f1_improve
            ]

    def parse_result_file(self, foldername: str) -> Tuple[Union[float, int]]:
        base_p, base_r, base_f1 = 0, 0, 0
        best_epoch, best_p, best_r, best_f1 = 0, 0, 0, 0
        for i, line in enumerate(FileIO.read_text(f'{foldername}/results.txt')):
            if i == 0:
                continue
            elif i == 1:
                base_p, base_r, _, base_f1 = extract_nums_from_string(line)
            else:
                epoch, p, r, _, f1 = extract_nums_from_string(line)
                if f1 > best_f1:
                    best_epoch, best_p, best_r, best_f1 = epoch, p, r, f1
        return base_p, base_r, base_f1, best_epoch, best_p, best_r, best_f1

    def save(self, filename: str) -> None:
        self.results_df.to_csv(f'{self.results_dir}/{filename}')

    def load(self, filename: str) -> None:
        self.results_df = pd.read_csv(f'{self.results_dir}/{filename}')
