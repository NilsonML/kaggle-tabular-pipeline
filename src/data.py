"""Загрузка данных, определение типа задачи и подготовка признаков."""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    X: pd.DataFrame
    y: np.ndarray
    X_test: pd.DataFrame
    test_ids: pd.Series
    sample_sub: pd.DataFrame
    id_col: str
    target: str
    task: str                      # binary | multiclass | regression
    cat_cols: list = field(default_factory=list)
    classes: np.ndarray | None = None  # исходные метки классов (для обратного декодирования)


def is_textual(s: pd.Series) -> bool:
    """Строковая/категориальная колонка (учитывает и object, и новый тип str в pandas 3)."""
    return (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
            or isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(s))


def detect_task(y: pd.Series) -> str:
    n_unique = y.nunique()
    if is_textual(y):
        return "binary" if n_unique == 2 else "multiclass"
    if n_unique == 2:
        return "binary"
    if pd.api.types.is_integer_dtype(y) and n_unique <= 20:
        return "multiclass"
    return "regression"


def load(cfg: dict) -> Dataset:
    data_dir = Path(cfg["data_dir"])
    train = pd.read_csv(data_dir / cfg.get("train_file", "train.csv"))
    test = pd.read_csv(data_dir / cfg.get("test_file", "test.csv"))
    sample = pd.read_csv(data_dir / cfg.get("sample_submission_file", "sample_submission.csv"))

    # id — первая колонка sample_submission, target — вторая (если не задано явно)
    id_col = cfg.get("id_col") or sample.columns[0]
    target = cfg.get("target") or sample.columns[1]
    if target not in train.columns:
        raise ValueError(
            f"Колонки '{target}' нет в train.csv. Укажи target в config.yaml "
            f"(в sample_submission колонки: {list(sample.columns)})"
        )

    task = cfg.get("task", "auto")
    if task == "auto":
        task = detect_task(train[target])

    y_raw = train[target]
    classes = None
    if task in ("binary", "multiclass"):
        classes, y = np.unique(y_raw.astype(str) if is_textual(y_raw) else y_raw, return_inverse=True)
    else:
        y = y_raw.to_numpy(dtype=float)
        if cfg.get("metric") == "rmsle":
            y = np.log1p(y)

    drop = [c for c in [id_col, target] + cfg.get("drop_cols", []) if c in train.columns]
    X = train.drop(columns=drop)
    X_test = test.drop(columns=[c for c in drop if c in test.columns])
    X_test = X_test[X.columns]

    X, X_test, cat_cols = prepare_features(X, X_test, cfg)

    return Dataset(X, y, X_test, test[id_col], sample, id_col, target, task, cat_cols, classes)


def prepare_features(X: pd.DataFrame, X_test: pd.DataFrame, cfg: dict):
    """Категориальные признаки → тип category с общими уровнями для train и test."""
    cat_cols = list(cfg.get("cat_cols") or [])
    for col in X.columns:
        if is_textual(X[col]):
            if col not in cat_cols:
                cat_cols.append(col)
        # целочисленные колонки с малым числом значений тоже часто категории
        elif cfg.get("int_as_cat_max_unique", 0) and pd.api.types.is_integer_dtype(X[col]) \
                and X[col].nunique() <= cfg["int_as_cat_max_unique"] and col not in cat_cols:
            cat_cols.append(col)

    for col in cat_cols:
        all_vals = pd.concat([X[col], X_test[col]]).astype(str)
        cats = pd.Categorical(all_vals).categories
        X[col] = pd.Categorical(X[col].astype(str), categories=cats)
        X_test[col] = pd.Categorical(X_test[col].astype(str), categories=cats)

    return X, X_test, cat_cols
