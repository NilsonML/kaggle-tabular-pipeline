"""Обёртки над LightGBM, CatBoost и XGBoost с единым интерфейсом fit/predict."""
import numpy as np
import pandas as pd


def _proba_out(model_proba: np.ndarray, task: str) -> np.ndarray:
    return model_proba[:, 1] if task == "binary" else model_proba


class LGBMModel:
    name = "lgbm"

    def __init__(self, task, n_classes, params, seed):
        import lightgbm as lgb
        self.lgb, self.task = lgb, task
        base = dict(n_estimators=5000, learning_rate=0.03, num_leaves=63, subsample=0.8,
                    subsample_freq=1, colsample_bytree=0.7, random_state=seed, verbose=-1)
        base.update(params or {})
        self.model = lgb.LGBMRegressor(**base) if task == "regression" else lgb.LGBMClassifier(**base)

    def fit(self, X, y, X_val, y_val, cat_cols):
        self.model.fit(X, y, eval_set=[(X_val, y_val)],
                       callbacks=[self.lgb.early_stopping(200, verbose=False)])
        return self

    def predict(self, X):
        if self.task == "regression":
            return self.model.predict(X)
        return _proba_out(self.model.predict_proba(X), self.task)

    def importance(self, columns):
        return pd.Series(self.model.feature_importances_, index=columns)


class CatBoostModel:
    name = "catboost"

    def __init__(self, task, n_classes, params, seed):
        from catboost import CatBoostClassifier, CatBoostRegressor
        self.task = task
        base = dict(iterations=5000, learning_rate=0.05, depth=6, random_seed=seed,
                    verbose=0, early_stopping_rounds=200, allow_writing_files=False)
        if task == "multiclass":
            base["loss_function"] = "MultiClass"
        base.update(params or {})
        self.model = CatBoostRegressor(**base) if task == "regression" else CatBoostClassifier(**base)

    @staticmethod
    def _prep(X, cat_cols):
        # CatBoost хочет строки в категориальных колонках
        X = X.copy()
        for c in cat_cols:
            X[c] = X[c].astype(str)
        return X

    def fit(self, X, y, X_val, y_val, cat_cols):
        self.cat_cols = cat_cols
        self.model.fit(self._prep(X, cat_cols), y, cat_features=cat_cols,
                       eval_set=(self._prep(X_val, cat_cols), y_val))
        return self

    def predict(self, X):
        X = self._prep(X, self.cat_cols)
        if self.task == "regression":
            return self.model.predict(X)
        return _proba_out(self.model.predict_proba(X), self.task)

    def importance(self, columns):
        return pd.Series(self.model.get_feature_importance(), index=columns)


class XGBModel:
    name = "xgb"

    def __init__(self, task, n_classes, params, seed):
        import xgboost as xgb
        self.task = task
        base = dict(n_estimators=5000, learning_rate=0.03, max_depth=7, subsample=0.8,
                    colsample_bytree=0.7, tree_method="hist", enable_categorical=True,
                    early_stopping_rounds=200, random_state=seed, verbosity=0)
        base.update(params or {})
        self.model = xgb.XGBRegressor(**base) if task == "regression" else xgb.XGBClassifier(**base)

    def fit(self, X, y, X_val, y_val, cat_cols):
        self.model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
        return self

    def predict(self, X):
        if self.task == "regression":
            return self.model.predict(X)
        return _proba_out(self.model.predict_proba(X), self.task)

    def importance(self, columns):
        return pd.Series(self.model.feature_importances_, index=columns)


MODELS = {cls.name: cls for cls in (LGBMModel, CatBoostModel, XGBModel)}
