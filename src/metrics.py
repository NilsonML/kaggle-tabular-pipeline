"""Метрики соревнований и правила, в каком виде отдавать предсказания."""
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn import metrics as skm


def _rmse(y, p):
    return float(np.sqrt(skm.mean_squared_error(y, p)))


def _map_at_3(y, proba):
    """MAP@3: часто используется в мультиклассовых Playground-соревнованиях."""
    top3 = np.argsort(-proba, axis=1)[:, :3]
    score = 0.0
    for k in range(3):
        score += (top3[:, k] == y).mean() / (k + 1)
    return float(score)


@dataclass
class Metric:
    name: str
    fn: Callable
    greater_is_better: bool
    needs_proba: bool  # True — метрика считается по вероятностям


METRICS = {
    "auc": Metric("auc", lambda y, p: skm.roc_auc_score(y, p), True, True),
    "logloss": Metric("logloss", lambda y, p: skm.log_loss(y, p), False, True),
    "accuracy": Metric("accuracy", lambda y, p: skm.accuracy_score(y, p), True, False),
    "f1_macro": Metric("f1_macro", lambda y, p: skm.f1_score(y, p, average="macro"), True, False),
    "map3": Metric("map3", _map_at_3, True, True),
    "rmse": Metric("rmse", _rmse, False, False),
    "mae": Metric("mae", lambda y, p: skm.mean_absolute_error(y, p), False, False),
    # RMSLE: обучаемся на log1p(y), поэтому по факту считаем RMSE в лог-пространстве
    "rmsle": Metric("rmsle", _rmse, False, False),
}

DEFAULT_METRIC = {"binary": "auc", "multiclass": "logloss", "regression": "rmse"}


def get_metric(name: str, task: str) -> Metric:
    name = (name or "auto").lower()
    if name == "auto":
        name = DEFAULT_METRIC[task]
    if name not in METRICS:
        raise ValueError(f"Неизвестная метрика '{name}'. Доступны: {list(METRICS)}")
    return METRICS[name]


def score(metric: Metric, task: str, y_true, raw_pred) -> float:
    """raw_pred: вероятность класса 1 (binary), матрица вероятностей (multiclass), число (regression)."""
    pred = raw_pred
    if not metric.needs_proba and task == "binary":
        pred = (raw_pred >= 0.5).astype(int)
    elif not metric.needs_proba and task == "multiclass":
        pred = raw_pred.argmax(axis=1)
    return metric.fn(y_true, pred)
