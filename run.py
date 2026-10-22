"""Запуск: python run.py --config config.yaml"""
import argparse

import yaml

from src.data import load
from src.metrics import get_metric
from src.train import run


def main():
    parser = argparse.ArgumentParser(description="Универсальный пайплайн для табличных Kaggle-соревнований")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--models", nargs="+", help="переопределить список моделей, напр. --models lgbm")
    parser.add_argument("--folds", type=int, help="переопределить число фолдов (для быстрой проверки)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.models:
        cfg["models"] = args.models
    if args.folds:
        cfg["n_folds"] = args.folds

    ds = load(cfg)
    run(ds, cfg, get_metric(cfg.get("metric", "auto"), ds.task))


if __name__ == "__main__":
    main()
