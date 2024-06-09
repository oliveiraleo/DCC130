from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class ModelDataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: pl.DataFrame
    y_test: np.ndarray


@dataclass
class Results:
    model: str
    accuracy_score: float
    precision_score: float
    recall_score: float
    f1_score: float
    train_num_rows: int
    test_num_rows: int
    num_features: int
    duration_training: float
    feature_selection: bool
