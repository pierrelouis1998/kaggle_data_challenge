import pickle
from pathlib import Path
from typing import Tuple, List

from networkx import Graph


def load_my_data(data_dir=Path("data")) -> Tuple[List[Graph], List[str], List[Graph]]:
    """Load dataset"""
    with open(data_dir / "training_data.pkl", 'rb') as f:
        train_data = pickle.load(f)
    with open(data_dir / "training_labels.pkl", 'rb') as f:
        train_labels = pickle.load(f)
    with open(data_dir / "test_data.pkl", 'rb') as f:
        test_data = pickle.load(f)

    return train_data, train_labels, test_data
