from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer


class TwitterPreprocessor:
    @staticmethod
    def preprocess(texts: List[str]) -> str:
        '''Preprocess text (username and link placeholders)'''
        clean_texts = []
        for text in texts:
            clean_text = []
            for t in text.split(" "):
                t = '@user' if t.startswith('@') and len(t) > 1 else t
                t = 'http' if t.startswith('http') else t
                clean_text.append(t)
            clean_texts.append(" ".join(clean_text))
        return clean_texts

    @staticmethod
    def check_input(texts: Union[str, List[str]]):
        '''Handle single text case'''
        if isinstance(texts, str):
            texts = [texts]
        return texts

    @staticmethod
    def run(texts: Union[str, List[str]]):
        texts = TwitterPreprocessor.check_input(texts)
        texts = TwitterPreprocessor.preprocess(texts)
        return texts


class TweetEmbedder:
    def __init__(self, model_name: str, device: str) -> None:
        super(TweetEmbedder, self).__init__()
        self.encoder = SentenceTransformer(model_name).to(device)

    def encode(self, texts: Union[List[str], str]) -> np.array:
        if not texts:
            return []
        texts = TwitterPreprocessor.run(texts)
        return self.encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
