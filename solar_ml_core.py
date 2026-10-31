"""Shared core for the LOSO experiments: configuration, data loading, models, evaluation."""

from __future__ import annotations

import os
import sys
import time
import logging
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
log = logging.getLogger("solar_ml")

REPO_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))


def section(title: str, char: str = "=", width: int = 72) -> None:
    log.info(char * width)
    log.info(title)
    log.info(char * width)


@dataclass(frozen=True)
class ExperimentConfig:

    base_dir: str = os.path.join(REPO_ROOT, "data", "processed")
    loso_folder_name: str = "LOSO_Folds"
    results_folder_name: str = "main"

    time_col: str = "utctime"
    site_col: str = "Site_ID"
    target_col: str = "Power_PU"
    underperforming_col: str = "Underperforming"

    base_features: tuple = (
        "HOUR_SIN", "HOUR_COS", "MONTH_SIN", "MONTH_COS",
        "GHI", "Temp_Ambient", "Wind_Speed", "Humidity",
        "Latitude", "Longitude", "Tilt",
    )
    poa_feature: str = "Irradiance_POA"
    cyclic_features: tuple = ("HOUR_SIN", "HOUR_COS", "MONTH_SIN", "MONTH_COS")

    lag_hours: tuple = (1, 2, 3, 24)
    lag_source_cols: tuple = ("Irradiance_POA", "GHI", "Temp_Ambient")
    lag_algorithms: tuple = ("XGBoost", "LSTM")

    tune_algorithms: tuple = ("KNN", "SVR")

    n_inner_validation_sites: int = 0

    tuning_subsample: int | None = 8000

    param_grids: dict = field(default_factory=lambda: {
        "MLR": [{}],
        "KNN": [{"n_neighbors": k, "weights": w}
                for k in (15, 50, 150) for w in ("uniform", "distance")],
        "SVR": [{"C": c, "gamma": g} for c in (1.0, 10.0, 100.0) for g in ("scale", 0.01)],
        "RandomForest": [{"n_estimators": 200, "min_samples_leaf": m} for m in (1, 5, 20)],
        "XGBoost": [{"max_depth": d, "learning_rate": lr}
                    for d in (4, 6, 10) for lr in (0.05, 0.15)],
        "LSTM": [{"lstm_units": u, "learning_rate": lr}
                 for u in (32, 64) for lr in (0.001, 0.01)],
    })

    scaled_algorithms: tuple = ("KNN", "SVR", "LSTM")

    lstm_lookback: int = 24
    lstm_units: int = 64
    lstm_epochs: int = 60
    lstm_batch_size: int = 256
    lstm_max_gap_fill_hours: int = 3

    daytime_ghi_threshold: float = 20.0
    mape_min_power_pu: float = 0.05

    permutation_repeats: int = 3
    permutation_max_samples: int = 3000

    train_subsample: int | None = None

    underperforming_modes: tuple = ("keep",)

    random_seed: int = 42

    include_poa: bool = True

    @property
    def loso_folder_path(self) -> str:
        return os.path.join(self.base_dir, self.loso_folder_name)

    @property
    def results_dir(self) -> str:
        return os.path.join(REPO_ROOT, "results", self.results_folder_name)

    @property
    def predictions_dir(self) -> str:
        return os.path.join(self.results_dir, "predictions")

    @property
    def models_dir(self) -> str:
        return os.path.join(self.results_dir, "trained_models")


CFG = ExperimentConfig()
for _d in (CFG.results_dir, CFG.predictions_dir, CFG.models_dir):
    os.makedirs(_d, exist_ok=True)
np.random.seed(CFG.random_seed)


def load_loso_folds(cfg: ExperimentConfig = CFG) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    section("Loading LOSO folds")
    folder = cfg.loso_folder_path
    sites = sorted(f.replace("LOSO_", "").replace("_train.csv", "")
                   for f in os.listdir(folder) if f.endswith("_train.csv"))

    folds: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for site in sites:
        tr = pd.read_csv(os.path.join(folder, f"LOSO_{site}_train.csv"), parse_dates=[cfg.time_col])
        te = pd.read_csv(os.path.join(folder, f"LOSO_{site}_test.csv"), parse_dates=[cfg.time_col])

        for name, df in (("train", tr), ("test", te)):
            n = len(df)
            df.drop_duplicates(subset=[cfg.site_col, cfg.time_col], keep="first", inplace=True)
            if len(df) != n:
                log.warning("[%s|%s] %d duplicate rows removed.", site, name, n - len(df))

        tr = tr.sort_values([cfg.site_col, cfg.time_col]).reset_index(drop=True)
        te = te.sort_values(cfg.time_col).reset_index(drop=True)

        if cfg.underperforming_col not in tr.columns:
            log.warning("[%s] column %s not found.",
                        site, cfg.underperforming_col)

        folds[site] = (tr, te)
        flagged = (te[cfg.underperforming_col].mean() * 100
                   if cfg.underperforming_col in te.columns else 0.0)
        log.info("[%-10s] train %s rows (%d stations) | test %s rows | flagged %%%.1f",
                 site, f"{len(tr):,}", tr[cfg.site_col].nunique(), f"{len(te):,}", flagged)

    if not folds:
        raise RuntimeError(f"No folds found in: {folder}")
    log.info("%d folds.", len(folds))
    return folds


def add_lag_features(df: pd.DataFrame, cfg: ExperimentConfig,
                     cols: tuple[str, ...], lags: tuple[int, ...]) -> pd.DataFrame:
    df = df.sort_values([cfg.site_col, cfg.time_col]).copy()
    for col in cols:
        for lag in lags:
            df[f"{col}_lag{lag}h"] = df.groupby(cfg.site_col)[col].shift(lag)
    return df


def lag_feature_names(cfg: ExperimentConfig = CFG) -> list[str]:
    return [f"{c}_lag{l}h" for c in cfg.lag_source_cols for l in cfg.lag_hours]


def build_feature_list(cfg: ExperimentConfig, poa_included: bool, algorithm: str) -> list[str]:
    features = list(cfg.base_features)
    if poa_included:
        features.append(cfg.poa_feature)
    if algorithm in cfg.lag_algorithms:
        features += lag_feature_names(cfg)
    return features


def scale_if_needed(train_df, test_df, features, algorithm, cfg):
    from sklearn.preprocessing import StandardScaler

    X_tr = train_df[features].to_numpy(dtype=float)
    X_te = test_df[features].to_numpy(dtype=float)
    if algorithm not in cfg.scaled_algorithms:
        return X_tr, X_te, None

    mask = np.array([f not in cfg.cyclic_features for f in features])
    scaler = StandardScaler().fit(X_tr[:, mask])
    X_tr, X_te = X_tr.copy(), X_te.copy()
    X_tr[:, mask] = scaler.transform(X_tr[:, mask])
    X_te[:, mask] = scaler.transform(X_te[:, mask])
    return X_tr, X_te, scaler


def build_model(algorithm: str, cfg: ExperimentConfig = CFG, params: dict | None = None):
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR

    p = dict(params or {})
    if algorithm == "MLR":
        return LinearRegression()
    if algorithm == "RandomForest":
        return RandomForestRegressor(**{"n_estimators": 300, "min_samples_leaf": 2, **p},
                                     n_jobs=-1, random_state=cfg.random_seed)
    if algorithm == "KNN":
        return KNeighborsRegressor(**{"n_neighbors": 15, "weights": "distance", **p}, n_jobs=-1)
    if algorithm == "SVR":
        return SVR(kernel="rbf", **{"C": 10.0, "epsilon": 0.01, "gamma": "scale", **p})
    if algorithm == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(**{"n_estimators": 400, "max_depth": 6, "learning_rate": 0.05,
                               "subsample": 0.8, "colsample_bytree": 0.8, **p},
                            random_state=cfg.random_seed, n_jobs=-1, verbosity=0)
    raise ValueError(f"Bilinmeyen algoritma: {algorithm}")


def reindex_hourly_with_limited_fill(site_df, cfg, feature_cols):
    site = site_df[cfg.site_col].iloc[0]
    df = site_df.set_index(cfg.time_col).sort_index()
    df = df[~df.index.duplicated(keep="first")]

    full = pd.date_range(df.index.min(), df.index.max(), freq="h", tz=df.index.tz)
    n_missing, n_total = len(full) - len(df), len(full)

    df = df.reindex(full)
    df[list(feature_cols)] = df[list(feature_cols)].interpolate(
        method="linear", limit=cfg.lstm_max_gap_fill_hours, limit_direction="both")
    if cfg.underperforming_col in df.columns:
        df[cfg.underperforming_col] = df[cfg.underperforming_col].fillna(True).astype(bool)
    df[cfg.site_col] = site
    df.index.name = cfg.time_col
    return df.reset_index(), n_missing, n_total


_QUIET: dict = {"on": False}


def build_lstm_dataset(df, features, cfg):
    Xs, ys, fs, ds = [], [], [], []
    has_flag = cfg.underperforming_col in df.columns

    for site, sdf in df.groupby(cfg.site_col):
        sdf = sdf.sort_values(cfg.time_col).reset_index(drop=True)
        rdf, n_missing, n_total = reindex_hourly_with_limited_fill(sdf, cfg, features)

        X = rdf[features].to_numpy(dtype=float)
        y = rdf[cfg.target_col].to_numpy(dtype=float)
        flag = (rdf[cfg.underperforming_col].to_numpy(dtype=bool) if has_flag
                else np.zeros(len(rdf), dtype=bool))
        day = (rdf["GHI_raw"].to_numpy(dtype=float) > cfg.daytime_ghi_threshold
               if "GHI_raw" in rdf.columns else np.ones(len(rdf), dtype=bool))

        lb = cfg.lstm_lookback
        for end in range(lb, len(X)):
            w, t = X[end - lb:end], y[end]
            if np.isnan(w).any() or np.isnan(t):
                continue
            Xs.append(w); ys.append(t); fs.append(flag[end]); ds.append(day[end])

        if not _QUIET.get("on"):
            log.info("[%s] hours %s | missing %s (%%%.1f) | valid windows %s",
                     site, f"{n_total:,}", f"{n_missing:,}",
                     100 * n_missing / n_total if n_total else 0, f"{len(Xs):,}")

    if not Xs:
        n = len(features)
        return (np.empty((0, cfg.lstm_lookback, n)), np.empty((0,)),
                np.empty((0,), bool), np.empty((0,), bool))
    return np.stack(Xs), np.array(ys), np.array(fs, bool), np.array(ds, bool)


def build_lstm_model(n_features: int, cfg: ExperimentConfig = CFG,
                     params: dict | None = None):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    p = dict(params or {})
    units = p.get("lstm_units", cfg.lstm_units)
    lr = p.get("learning_rate", 0.001)

    tf.random.set_seed(cfg.random_seed)
    model = models.Sequential([
        layers.Input(shape=(cfg.lstm_lookback, n_features)),
        layers.LSTM(units),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="linear"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="mse", metrics=["mae"])
    return model


def compute_metrics(y_true, y_pred, cfg, daytime_mask=None) -> dict:
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    m = {"R2": float(r2_score(y_true, y_pred)),
         "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
         "MAE": float(mean_absolute_error(y_true, y_pred))}

    mask = y_true > cfg.mape_min_power_pu
    if daytime_mask is not None:
        mask = mask & daytime_mask

    if mask.sum():
        yt, yp = y_true[mask], y_pred[mask]
        m["MAPE_daytime"] = float(np.mean(np.abs((yt - yp) / yt)) * 100)
        m["sMAPE_daytime"] = float(np.mean(2 * np.abs(yt - yp) / (np.abs(yt) + np.abs(yp))) * 100)
        m["n_mape_points"] = int(mask.sum())
    else:
        m["MAPE_daytime"] = m["sMAPE_daytime"] = np.nan
        m["n_mape_points"] = 0
    return m


def permutation_feature_importance(predict_fn: Callable, X, y_true, feature_names,
                                   is_sequence=False, n_repeats=3, seed=42,
                                   max_samples=None) -> dict[str, float]:
    from sklearn.metrics import r2_score

    rng = np.random.default_rng(seed)
    if max_samples and X.shape[0] > max_samples:
        idx = rng.choice(X.shape[0], size=max_samples, replace=False)
        X, y_true = X[idx], y_true[idx]

    base = r2_score(y_true, predict_fn(X))
    out = {}
    for i, feat in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            p = rng.permutation(X.shape[0])
            if is_sequence:
                Xp[:, :, i] = Xp[p][:, :, i]
            else:
                Xp[:, i] = Xp[p, i]
            drops.append(base - r2_score(y_true, predict_fn(Xp)))
        out[feat] = float(np.mean(drops))
    return out


def run_single_fold(algorithm, site, train_df, test_df, exclude_flagged: bool,
                    cfg: ExperimentConfig = CFG, include_poa: bool | None = None,
                    params: dict | None = None, quiet: bool = False,
                    return_score_only: bool = False):
    import joblib

    include_poa = cfg.include_poa if include_poa is None else include_poa
    tag = "_exclude" if exclude_flagged else ""
    features = build_feature_list(cfg, include_poa, algorithm)
    flag_col = cfg.underperforming_col
    t0 = time.time()

    if not quiet:
        log.info("%s | test=%s%s%s", algorithm, site,
                 " | flagged hours excluded" if exclude_flagged else "",
                 f" | {params}" if params else "")

    train_df, test_df = train_df.copy(), test_df.copy()
    if algorithm in cfg.lag_algorithms:
        train_df = add_lag_features(train_df, cfg, cfg.lag_source_cols, cfg.lag_hours)
        test_df = add_lag_features(test_df, cfg, cfg.lag_source_cols, cfg.lag_hours)
        train_df = train_df.dropna(subset=features + [cfg.target_col])
        test_df = test_df.dropna(subset=features + [cfg.target_col])

    if algorithm == "LSTM":
        X_tr_s, X_te_s, _ = scale_if_needed(train_df, test_df, features, algorithm, cfg)
        tr_s, te_s = train_df.copy(), test_df.copy()
        tr_s[features], te_s[features] = X_tr_s, X_te_s
        tr_s["GHI_raw"] = train_df["GHI"].to_numpy()
        te_s["GHI_raw"] = test_df["GHI"].to_numpy()

        Xtr, ytr, ftr, _ = build_lstm_dataset(tr_s, features, cfg)
        Xte, yte, fte, dte = build_lstm_dataset(te_s, features, cfg)
        if len(Xte) == 0:
            log.warning("No valid windows, skipped.")
            return {}, {}

        if cfg.train_subsample and len(Xtr) > cfg.train_subsample:
            idx = np.random.default_rng(cfg.random_seed).choice(
                len(Xtr), size=cfg.train_subsample, replace=False)
            Xtr, ytr, ftr = Xtr[idx], ytr[idx], ftr[idx]

        w = np.where(ftr, 0.0, 1.0) if exclude_flagged else np.ones(len(ytr))

        model = build_lstm_model(len(features), cfg, params)
        _t = time.time()
        model.fit(Xtr, ytr, sample_weight=w, epochs=cfg.lstm_epochs,
                  batch_size=cfg.lstm_batch_size, verbose=0)
        fit_s = time.time() - _t

        _t = time.time()
        y_pred = model.predict(Xte, verbose=0).ravel()
        pred_s = time.time() - _t
        y_true = yte
        n_train = len(ytr)
        eval_mask = ~fte if exclude_flagged else np.ones(len(yte), bool)
        daytime = dte
        if return_score_only:
            return compute_metrics(y_true[eval_mask], y_pred[eval_mask], cfg,
                                   dte[eval_mask])["R2"]

        model.save(os.path.join(cfg.models_dir, f"LSTM_{site}{tag}.keras"))

        importance = permutation_feature_importance(
            lambda X: model.predict(X, verbose=0).ravel(), Xte[eval_mask], y_true[eval_mask],
            features, True, cfg.permutation_repeats, cfg.random_seed, cfg.permutation_max_samples)

    else:
        if exclude_flagged and flag_col in train_df.columns:
            n = len(train_df)
            train_df = train_df[~train_df[flag_col].astype(bool)]
            log.info("%s flagged rows removed from training.", f"{n - len(train_df):,}")

        X_tr, X_te, scaler = scale_if_needed(train_df, test_df, features, algorithm, cfg)
        y_train = train_df[cfg.target_col].to_numpy(dtype=float)
        y_true = test_df[cfg.target_col].to_numpy(dtype=float)

        if cfg.train_subsample and len(X_tr) > cfg.train_subsample:
            idx = np.random.default_rng(cfg.random_seed).choice(
                len(X_tr), size=cfg.train_subsample, replace=False)
            log.info("Training subsampled to %s rows.",
                     f"{cfg.train_subsample:,}")
            X_tr, y_train = X_tr[idx], y_train[idx]

        model = build_model(algorithm, cfg, params)
        _t = time.time()
        model.fit(X_tr, y_train)
        fit_s = time.time() - _t

        _t = time.time()
        y_pred = model.predict(X_te)
        pred_s = time.time() - _t
        n_train = len(y_train)

        daytime = (test_df["GHI"] > cfg.daytime_ghi_threshold).to_numpy()
        eval_mask = (~test_df[flag_col].astype(bool).to_numpy()
                     if exclude_flagged and flag_col in test_df.columns
                     else np.ones(len(y_true), bool))

        if return_score_only:
            daytime = (test_df["GHI"] > cfg.daytime_ghi_threshold).to_numpy()
            return compute_metrics(y_true[eval_mask], y_pred[eval_mask], cfg,
                                   daytime[eval_mask])["R2"]

        joblib.dump({"model": model, "scaler": scaler, "features": features},
                    os.path.join(cfg.models_dir, f"{algorithm}_{site}{tag}.joblib"))

        importance = permutation_feature_importance(
            model.predict, X_te[eval_mask], y_true[eval_mask], features,
            False, cfg.permutation_repeats, cfg.random_seed, cfg.permutation_max_samples)

    metrics = compute_metrics(y_true[eval_mask], y_pred[eval_mask], cfg, daytime[eval_mask])
    if return_score_only:
        return metrics["R2"]

    pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "used_in_metrics": eval_mask}).to_csv(
        os.path.join(cfg.predictions_dir, f"{algorithm}_{site}{tag}.csv"), index=False)

    result = {"Model": algorithm, "Test_Station": site,
              "n_train_points": int(n_train), "n_eval_points": int(eval_mask.sum()),
              "Fit_Seconds": round(fit_s, 3), "Predict_Seconds": round(pred_s, 3),
              "Fold_Seconds": round(time.time() - t0, 1), **metrics}
    if exclude_flagged:
        result["Underperf_Mode"] = "exclude"
    log.info("R2=%.3f  RMSE=%.4f  MAE=%.4f | fit %.1fs  predict %.2fs  (fold %.1fs)",
             metrics["R2"], metrics["RMSE"], metrics["MAE"], fit_s, pred_s, time.time() - t0)
    return result, importance


def tune_hyperparameters(algorithm, train_df, cfg: ExperimentConfig = CFG,
                         include_poa: bool | None = None):
    grid = cfg.param_grids.get(algorithm, [{}])
    if algorithm not in cfg.tune_algorithms or len(grid) <= 1:
        return {}, None

    sites = sorted(train_df[cfg.site_col].unique())
    n = cfg.n_inner_validation_sites
    val_sites = sites if n <= 0 else sites[-max(1, min(n, len(sites) - 1)):]

    log.info("%s tuning | %d combinations x %d validation stations%s",
             algorithm, len(grid), len(val_sites),
             f" | subsample {cfg.tuning_subsample:,}" if cfg.tuning_subsample else "")

    _QUIET["on"] = True
    scores = []
    try:
        for params in grid:
            fold_r2 = []
            for vs in val_sites:
                inner_tr = train_df[train_df[cfg.site_col] != vs]
                inner_va = train_df[train_df[cfg.site_col] == vs]
                if cfg.tuning_subsample and len(inner_tr) > cfg.tuning_subsample:
                    inner_tr = inner_tr.sample(cfg.tuning_subsample,
                                               random_state=cfg.random_seed)
                try:
                    fold_r2.append(run_single_fold(algorithm, vs, inner_tr, inner_va,
                                                   False, cfg, include_poa, params,
                                                   quiet=True, return_score_only=True))
                except Exception as exc:
                    log.warning("  failed: %s @ %s -> %s", params, vs, exc)
            if not fold_r2:
                continue
            mean_r2 = float(np.mean(fold_r2))
            scores.append((mean_r2, params))
            log.info("  %-46s R2=%.4f  (+/-%.3f, %d stations)",
                     str(params), mean_r2, float(np.std(fold_r2)), len(fold_r2))
    finally:
        _QUIET["on"] = False

    if not scores:
        log.warning("Tuning failed; default parameters used.")
        return {}, None

    best_r2, best = max(scores, key=lambda x: x[0])
    log.info("Best: %s  (inner R2 = %.4f)", best, best_r2)
    return best, best_r2


def run_algorithm(algorithm: str, folds: dict, cfg: ExperimentConfig = CFG,
                  underperf_modes=None, include_poa: bool | None = None,
                  tune: bool = True):
    underperf_modes = underperf_modes or cfg.underperforming_modes

    results, importances = [], []
    for mode in underperf_modes:
        section(f"{algorithm} | {len(folds)} folds"
                + (" | flagged hours excluded" if mode == "exclude" else ""))
        for site, (tr, te) in folds.items():
            params, inner_r2 = ({}, None)
            if tune:
                params, inner_r2 = tune_hyperparameters(algorithm, tr, cfg, include_poa)

            r, imp = run_single_fold(algorithm, site, tr, te, mode == "exclude",
                                     cfg, include_poa, params)
            if r:
                r["Params"] = str(params) if params else "default"
                r["Inner_R2"] = inner_r2
                results.append(r)
                rows = [{"Model": algorithm, "Test_Station": site,
                         "Feature": f, "Importance": v} for f, v in imp.items()]
                if mode == "exclude":
                    for r_ in rows:
                        r_["Underperf_Mode"] = "exclude"
                importances += rows

    res_df, imp_df = pd.DataFrame(results), pd.DataFrame(importances)
    res_df.to_csv(os.path.join(cfg.results_dir, f"results_{algorithm}.csv"), index=False)
    imp_df.to_csv(os.path.join(cfg.results_dir, f"importance_{algorithm}.csv"), index=False)
    log.info("Saved results_%s.csv and importance_%s.csv", algorithm, algorithm)
    return res_df, imp_df


def summarize(algorithm: str, res_df: pd.DataFrame, imp_df: pd.DataFrame, cfg=CFG):
    metric_cols = ["R2", "RMSE", "MAE", "MAPE_daytime", "sMAPE_daytime"]
    section(f"{algorithm} summary")

    has_mode = "Underperf_Mode" in res_df.columns and res_df["Underperf_Mode"].notna().any()
    idx_cols = (["Underperf_Mode"] if has_mode else []) + ["Test_Station"]

    print("\nPer-fold results:")
    print(res_df[idx_cols + metric_cols].round(4).to_string(index=False))

    print("\nMean over folds:")
    if has_mode:
        print(res_df.groupby("Underperf_Mode")[metric_cols].mean().round(4).to_string())
    else:
        print(res_df[metric_cols].mean().round(4).to_string())

    if "Params" in res_df.columns and res_df["Params"].nunique() > 0:
        print("\nSelected parameters per fold:")
        cols = ["Test_Station", "Params"] + (["Inner_R2"] if "Inner_R2" in res_df else []) + ["R2"]
        print(res_df[cols].round(4).to_string(index=False))

    if "Fit_Seconds" in res_df.columns:
        print(f"\nFit time: mean {res_df['Fit_Seconds'].mean():.1f}s | "
              f"total {res_df['Fit_Seconds'].sum():.1f}s | "
              f"predict: mean {res_df['Predict_Seconds'].mean():.2f}s")

    if has_mode and res_df["Underperf_Mode"].nunique() > 1:
        print("\nEffect of excluding flagged hours on R2:")
        p = res_df.pivot_table(index="Test_Station", columns="Underperf_Mode", values="R2")
        p["Difference"] = p.get("exclude", np.nan) - p.get("keep", np.nan)
        print(p.round(4).to_string())

    if not imp_df.empty:
        print("\nTop 10 features (mean permutation importance):")
        print(imp_df.groupby("Feature")["Importance"].mean()
              .sort_values(ascending=False).head(10).round(4).to_string())
