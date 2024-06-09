import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

from dtos import ModelDataset, Results


def sample_rows(
    group_by: pl.dataframe.group_by.GroupBy, percentage: int
) -> pl.DataFrame:
    """
    Amostra linhas de um DataFrame em um objeto GroupBy.

    Parâmetros
    ----------
    group_by : pl.dataframe.group_by.GroupBy
        objeto GroupBy.
    percentage : float
        porcentagem de linhas a serem amostradas.

    Retorna
    -------
    pl.DataFrame
        DataFrame amostrado.
    """

    percentage /= 100
    dfs = []
    for _, df in group_by:
        if len(df) * percentage <= 1:
            dfs.append(df.sample(n=1))
            continue
        dfs.append(df.sample(fraction=percentage, seed=42))

    return pl.concat(dfs)


def generate_dataframe(file_list: list[str], percentage: int) -> pl.DataFrame:
    """
    Cria um DataFrame com uma amostra de linhas de cada arquivo CSV de uma lista.

    Parâmetros
    ----------
    file_list : list[str]
        lista de arquivos CSV.
    percentage : int
        porcentagem de linhas a serem amostradas.

    Retorna
    -------
    pl.DataFrame
        DataFrame amostrado.
    """

    dfs = []
    for file in file_list:
        df = pl.read_csv(file)
        dfs.append(
            sample_rows(
                group_by=df.group_by(["label"], maintain_order=True),
                percentage=percentage,
            )
        )
    return pl.concat(dfs)


def correlation(dataset: pl.DataFrame, threshold: float) -> list:
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix[i, j]) > threshold:
                col_name = corr_matrix.columns[i]
                col_corr.add(col_name)
    return list(col_corr)


def evaluate_model(model, data: ModelDataset, feature_selection_bool: bool):
    start_training = time.time()
    model.fit(data.x_train, data.y_train.ravel())
    end_training = time.time()

    y_pred = model.predict(data.x_test)

    return Results(
        model=type(model).__name__,
        accuracy_score=accuracy_score(data.y_test, y_pred),
        precision_score=precision_score(data.y_test, y_pred, average="macro"),
        recall_score=recall_score(data.y_test, y_pred, average="macro"),
        f1_score=f1_score(data.y_test, y_pred, average="macro"),
        train_num_rows=len(data.x_train),
        test_num_rows=len(data.x_test),
        num_features=len(np.unique(data.y_train)),
        duration_training=float(f"{end_training - start_training:.4f}"),
        feature_selection=feature_selection_bool,
    )


def save_results(results: Results, save_file: str):
    df = pl.DataFrame(vars(results))

    file_path = Path(save_file)

    if file_path.exists():
        df_already_exists = pl.read_csv(file_path)
        result = pl.concat([df_already_exists, df])
        result.write_csv(file_path)
    else:
        df.write_csv(file_path)
