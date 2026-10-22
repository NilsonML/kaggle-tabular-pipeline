"""Кросс-валидация, блендинг моделей и сборка файла сабмита."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold, StratifiedKFold

from .data import Dataset
from .metrics import Metric, score
from .models import MODELS


def run_cv(ds: Dataset, model_name: str, cfg: dict, metric: Metric):
    """Обучает модель на K фолдах. Возвращает OOF-предсказания, предсказания на test и скоры."""
    n_folds, seed = cfg.get("n_folds", 5), cfg.get("seed", 42)
    n_classes = len(ds.classes) if ds.classes is not None else 0
    splitter = (KFold(n_folds, shuffle=True, random_state=seed) if ds.task == "regression"
                else StratifiedKFold(n_folds, shuffle=True, random_state=seed))

    shape = (len(ds.X), n_classes) if ds.task == "multiclass" else (len(ds.X),)
    test_shape = (len(ds.X_test), n_classes) if ds.task == "multiclass" else (len(ds.X_test),)
    oof, test_pred = np.zeros(shape), np.zeros(test_shape)
    fold_scores, importances = [], []

    params = (cfg.get("model_params") or {}).get(model_name, {})
    for fold, (tr, va) in enumerate(splitter.split(ds.X, ds.y)):
        model = MODELS[model_name](ds.task, n_classes, params, seed + fold)
        model.fit(ds.X.iloc[tr], ds.y[tr], ds.X.iloc[va], ds.y[va], ds.cat_cols)
        oof[va] = model.predict(ds.X.iloc[va])
        test_pred += model.predict(ds.X_test) / n_folds
        s = score(metric, ds.task, ds.y[va], oof[va])
        fold_scores.append(s)
        importances.append(model.importance(ds.X.columns))
        print(f"  [{model_name}] fold {fold + 1}/{n_folds}: {metric.name} = {s:.5f}")

    total = score(metric, ds.task, ds.y, oof)
    print(f"  [{model_name}] OOF {metric.name} = {total:.5f} (± {np.std(fold_scores):.5f})")
    imp = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
    return oof, test_pred, {"oof": total, "folds": fold_scores}, imp


def find_blend_weights(oofs: dict, ds: Dataset, metric: Metric) -> dict:
    """Подбирает неотрицательные веса моделей (сумма = 1), оптимизируя метрику на OOF."""
    names = list(oofs)
    if len(names) == 1:
        return {names[0]: 1.0}

    def blend(w):
        w = np.abs(w) / np.abs(w).sum()
        return sum(wi * oofs[n] for wi, n in zip(w, names))

    def loss(w):
        s = score(metric, ds.task, ds.y, blend(w))
        return -s if metric.greater_is_better else s

    res = minimize(loss, np.ones(len(names)) / len(names), method="Nelder-Mead",
                   options={"maxiter": 300, "xatol": 1e-4, "fatol": 1e-6})
    w = np.abs(res.x) / np.abs(res.x).sum()

    # оптимизатор может застрять: если бленд хуже лучшей одиночной модели — берём её
    best = min(names, key=lambda n: loss(np.eye(len(names))[names.index(n)]))
    if loss(np.eye(len(names))[names.index(best)]) < loss(w):
        return {n: float(n == best) for n in names}
    return dict(zip(names, map(float, w)))


def make_submission(ds: Dataset, pred: np.ndarray, metric: Metric, cfg: dict) -> pd.DataFrame:
    sub = pd.DataFrame({ds.id_col: ds.test_ids})
    sample_cols = list(ds.sample_sub.columns)

    if ds.task == "regression":
        sub[ds.target] = np.expm1(pred) if cfg.get("metric") == "rmsle" else pred
    elif ds.task == "binary":
        sub[ds.target] = pred if metric.needs_proba else ds.classes[(pred >= 0.5).astype(int)]
    elif metric.name == "map3":
        top3 = np.argsort(-pred, axis=1)[:, :3]
        sub[ds.target] = [" ".join(map(str, ds.classes[row])) for row in top3]
    elif len(sample_cols) > 2:
        # сабмит вида id, class_A, class_B, ... — вероятность на каждый класс
        for col in sample_cols[1:]:
            idx = np.where(ds.classes.astype(str) == str(col).split("_")[-1])[0]
            idx = idx[0] if len(idx) else sample_cols[1:].index(col)
            sub[col] = pred[:, idx]
    else:
        sub[ds.target] = ds.classes[pred.argmax(axis=1)]
    return sub


def run(ds: Dataset, cfg: dict, metric: Metric) -> Path:
    out = Path(cfg.get("output_dir", "outputs")) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    print(f"Задача: {ds.task} | метрика: {metric.name} | train {ds.X.shape} | test {ds.X_test.shape}")
    print(f"Категориальные признаки: {ds.cat_cols or 'нет'}\n")

    oofs, tests, report = {}, {}, {}
    for name in cfg.get("models", ["lgbm", "catboost", "xgb"]):
        oof, test_pred, scores, imp = run_cv(ds, name, cfg, metric)
        oofs[name], tests[name], report[name] = oof, test_pred, scores
        np.save(out / f"oof_{name}.npy", oof)
        np.save(out / f"test_{name}.npy", test_pred)
        imp.to_csv(out / f"importance_{name}.csv", header=["importance"])
        make_submission(ds, test_pred, metric, cfg).to_csv(out / f"submission_{name}.csv", index=False)

    weights = find_blend_weights(oofs, ds, metric)
    blend_oof = sum(w * oofs[n] for n, w in weights.items())
    blend_test = sum(w * tests[n] for n, w in weights.items())
    report["blend"] = {"oof": score(metric, ds.task, ds.y, blend_oof), "weights": weights}
    make_submission(ds, blend_test, metric, cfg).to_csv(out / "submission_blend.csv", index=False)

    with open(out / "cv_report.json", "w") as f:
        json.dump({"task": ds.task, "metric": metric.name, "results": report}, f, indent=2, ensure_ascii=False)

    print("\nИтог по OOF:")
    for name, r in report.items():
        print(f"  {name:10s} {r['oof']:.5f}")
    print(f"  веса бленда: {', '.join(f'{n}={w:.2f}' for n, w in weights.items())}")
    print(f"\nРезультаты: {out}/  →  отправляй submission_blend.csv")
    return out
