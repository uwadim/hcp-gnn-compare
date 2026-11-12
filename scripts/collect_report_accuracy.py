#!/usr/bin/env python3
"""
Собрать accuracy для таблиц report_baselines_gnn.md из уже посчитанных данных.

Источники:
- top-n: кэш скоров + best_n_by_train (знак по train своего fold, порог 0);
- LogReg ``all``: кэш признаков + переfit StandardScaler→LogReg (как в baseline);
- SkipGCN stage 2 / smoke / CF-MSR smoke: MLflow ``outer0_test_accuracy`` и train/val/oof.

Результат: ``artifacts/report_accuracy/*.csv``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from batch_runner import OUTER_FOLDS, TASKS, VARIANTS  # noqa: E402
from graph_features_logreg import (  # noqa: E402
    LogRegConfig,
    fit_predict_logreg,
    load_table,
    parse_feature_set,
)
from mean_edge_baseline import fit_sign  # noqa: E402

log = logging.getLogger("collect_report_accuracy")

OUT_DIR = ROOT / "artifacts" / "report_accuracy"
TOP_N_CACHE = ROOT / "artifacts" / "baselines_cv" / "top_n" / "cache"
TOP_N_BEST = ROOT / "artifacts" / "baselines_cv" / "top_n" / "best_n_by_train.csv"
LOGREG_CACHE = ROOT / "artifacts" / "baselines_cv" / "logreg" / "cache"
LOGREG_CACHE_OUTER0 = ROOT / "artifacts" / "graph_features_logreg" / "cache"
TRACKING_URI = f"sqlite:///{ROOT / 'artifacts' / 'mlflow.db'}"


def _accuracy(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    """Accuracy при пороге ``threshold`` на скоре (класс 1 если score >= threshold)."""
    y = np.asarray(y_true).astype(int)
    pred = (np.asarray(scores) >= threshold).astype(int)
    if y.size == 0:
        raise ValueError("пустой набор для accuracy")
    return float((pred == y).mean())


def _train_accuracy_threshold(y_train: np.ndarray, scores: np.ndarray) -> float:
    """
    Порог на train, максимизирующий accuracy (класс 1 при score >= thr).

    Кандидаты — середины между соседними уникальными скорами. Нужен, потому что
    знаковый top-n скор не обязан центрироваться около 0 (в отличие от prob=0.5).
    """
    y = np.asarray(y_train).astype(int)
    s = np.asarray(scores, dtype=float)
    uniq = np.unique(s)
    if uniq.size == 1:
        # любой порог ниже/выше даёт константный класс — берём границу
        return float(uniq[0])
    mids = 0.5 * (uniq[:-1] + uniq[1:])
    candidates = np.concatenate(
        ([uniq[0] - abs(uniq[0]) - 1.0], mids, [uniq[-1] + abs(uniq[-1]) + 1.0])
    )
    best_thr = float(mids[0])
    best_acc = -1.0
    for thr in candidates:
        acc = _accuracy(y, s, float(thr))
        if acc > best_acc:
            best_acc = acc
            best_thr = float(thr)
    return best_thr


def _col_for_n(n: int, n_edges: int = 71631) -> str:
    return f"top{n_edges}" if int(n) == int(n_edges) else f"top{int(n)}"


def top_n_accuracy_pooled(folds: Sequence[int] = OUTER_FOLDS) -> pd.DataFrame:
    """Pooled accuracy top-n: знак и порог по train своего fold, n из best_n_by_train."""
    best = pd.read_csv(TOP_N_BEST)
    rows: List[dict] = []
    for _, row in best.iterrows():
        variant, task, n = str(row["variant"]), str(row["task"]), int(row["best_n"])
        col = _col_for_n(n)
        y_parts: List[np.ndarray] = []
        pred_parts: List[np.ndarray] = []
        for fold in folds:
            path = TOP_N_CACHE / f"{variant}_{task}_outer{fold}.csv"
            if not path.is_file():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path)
            if col not in frame.columns:
                candidates = [c for c in frame.columns if c.startswith("top")]
                raise KeyError(f"{path}: нет {col}, есть {candidates[-3:]}")
            train = frame[frame["role"] == "train"]
            test = frame[frame["role"] == "test"]
            sign, _ = fit_sign(train["label"].to_numpy(), train[col].to_numpy())
            train_s = sign * train[col].to_numpy()
            test_s = sign * test[col].to_numpy()
            thr = _train_accuracy_threshold(train["label"].to_numpy(), train_s)
            y_parts.append(test["label"].to_numpy())
            pred_parts.append((test_s >= thr).astype(int))
        y = np.concatenate(y_parts)
        pred = np.concatenate(pred_parts)
        rows.append(
            {
                "variant": variant,
                "task": task,
                "best_n": n,
                "accuracy": float((pred == y.astype(int)).mean()),
                "n_test": int(y.size),
                "folds": ",".join(str(k) for k in folds),
            }
        )
    return pd.DataFrame(rows)


def _load_logreg_table(variant: str, task: str, fold: int):
    """Загрузить кэш признаков (baselines_cv или outer0-only)."""
    pattern = f"{variant}_{task}_outer{fold}_*.npz"
    for cache_dir in (LOGREG_CACHE, LOGREG_CACHE_OUTER0):
        matches = sorted(cache_dir.glob(pattern)) if cache_dir.is_dir() else []
        if matches:
            table = load_table(matches[0])
            if table is not None:
                return table
    raise FileNotFoundError(f"Нет кэша LogReg для {variant}/{task}/outer{fold}")


def logreg_accuracy(
    folds: Sequence[int],
    feature_set: str = "all",
    seed: int = 42,
) -> pd.DataFrame:
    """Accuracy LogReg на кэше признаков (переfit, без чтения графов)."""
    config = LogRegConfig()
    groups = parse_feature_set(feature_set)
    rows: List[dict] = []
    rng = np.random.default_rng(seed)
    # GridSearchCV использует numpy RNG косвенно через sklearn; фиксируем seed процесса.
    for task in TASKS:
        y_all: List[np.ndarray] = []
        p_all: List[np.ndarray] = []
        for variant in VARIANTS:
            y_parts: List[np.ndarray] = []
            p_parts: List[np.ndarray] = []
            for fold in folds:
                table = _load_logreg_table(variant, task, fold)
                x_all = table.select(groups)
                is_train = (table.meta["role"] == "train").to_numpy()
                train = table.meta[is_train]
                test = table.meta[~is_train]
                # небольшой jitter seed между ячейками не нужен: C выбирается по GroupKFold
                _ = rng.integers(0, 1)  # keep rng used
                prob, best_c, _ = fit_predict_logreg(
                    x_all[is_train],
                    train["label"].to_numpy(),
                    train["subject"].tolist(),
                    x_all[~is_train],
                    config,
                    n_jobs=1,
                )
                y_parts.append(test["label"].to_numpy())
                p_parts.append(prob)
                if len(folds) == 1:
                    rows.append(
                        {
                            "variant": variant,
                            "task": task,
                            "accuracy": _accuracy(test["label"].to_numpy(), prob, 0.5),
                            "best_c": best_c,
                            "n_test": int(len(test)),
                            "folds": str(folds[0]),
                        }
                    )
            if len(folds) > 1:
                y = np.concatenate(y_parts)
                p = np.concatenate(p_parts)
                rows.append(
                    {
                        "variant": variant,
                        "task": task,
                        "accuracy": _accuracy(y, p, 0.5),
                        "best_c": float("nan"),
                        "n_test": int(y.size),
                        "folds": ",".join(str(k) for k in folds),
                    }
                )
            # silence unused when folds==1 path already appended
            del y_all, p_all
    return pd.DataFrame(rows)


def logreg_accuracy_fixed(folds: Sequence[int], feature_set: str = "all") -> pd.DataFrame:
    """Accuracy LogReg: на каждом fold отдельно, затем pooled по субъектам."""
    config = LogRegConfig()
    groups = parse_feature_set(feature_set)
    rows: List[dict] = []
    for variant in VARIANTS:
        for task in TASKS:
            y_parts: List[np.ndarray] = []
            p_parts: List[np.ndarray] = []
            cs: List[float] = []
            for fold in folds:
                table = _load_logreg_table(variant, task, fold)
                x_all = table.select(groups)
                is_train = (table.meta["role"] == "train").to_numpy()
                train = table.meta[is_train]
                test = table.meta[~is_train]
                prob, best_c, _ = fit_predict_logreg(
                    x_all[is_train],
                    train["label"].to_numpy(),
                    train["subject"].tolist(),
                    x_all[~is_train],
                    config,
                    n_jobs=1,
                )
                y_parts.append(test["label"].to_numpy())
                p_parts.append(prob)
                cs.append(best_c)
            y = np.concatenate(y_parts)
            p = np.concatenate(p_parts)
            rows.append(
                {
                    "variant": variant,
                    "task": task,
                    "accuracy": _accuracy(y, p, 0.5),
                    "best_c_mean": float(np.mean(cs)),
                    "n_test": int(y.size),
                    "folds": ",".join(str(k) for k in folds),
                }
            )
            log.info("logreg %s/%s folds=%s acc=%.4f", variant, task, folds, rows[-1]["accuracy"])
    return pd.DataFrame(rows)


def mlflow_test_accuracy(prefix: str, model: str) -> pd.DataFrame:
    """``outer0_test_accuracy`` parent-run по variant×task."""
    client = MlflowClient(TRACKING_URI)
    rows: List[dict] = []
    for variant in VARIANTS:
        for task in TASKS:
            name = f"{prefix}_{variant}_{task}_{model}"
            exp = client.get_experiment_by_name(name)
            if exp is None:
                raise RuntimeError(f"Нет experiment {name}")
            runs = client.search_runs(exp.experiment_id, max_results=100)
            parents = [
                r
                for r in runs
                if r.info.status == "FINISHED" and "outer0_test_accuracy" in r.data.metrics
            ]
            if not parents:
                raise RuntimeError(f"Нет outer0_test_accuracy в {name}")
            run = max(parents, key=lambda r: r.info.end_time or 0)
            m = run.data.metrics
            rows.append(
                {
                    "variant": variant,
                    "task": task,
                    "model": model,
                    "accuracy": float(m["outer0_test_accuracy"]),
                    "roc_auc": float(m.get("outer0_test_roc_auc", float("nan"))),
                    "oof_accuracy": float(m.get("oof_accuracy", float("nan"))),
                    "oof_roc_auc": float(m.get("oof_roc_auc", float("nan"))),
                }
            )
    return pd.DataFrame(rows)


def skipgcn_overfit_accuracy() -> pd.DataFrame:
    """Средние train/val/gap/oof/test accuracy по variant (9 fold-run + parent)."""
    client = MlflowClient(TRACKING_URI)
    rows: List[dict] = []
    for variant in VARIANTS:
        train_acc: List[float] = []
        val_acc: List[float] = []
        test_acc: List[float] = []
        oof_acc: List[float] = []
        epochs: List[float] = []
        for task in TASKS:
            name = f"hcp_stage2_compare_{variant}_{task}_skipgcn"
            exp = client.get_experiment_by_name(name)
            if exp is None:
                raise RuntimeError(name)
            runs = client.search_runs(exp.experiment_id, max_results=100)
            folds = [
                r
                for r in runs
                if r.info.status == "FINISHED" and r.info.run_name.startswith("outer")
            ]
            if len(folds) != 9:
                log.warning("%s: fold-run %d (ожидалось 9)", name, len(folds))
            for r in folds:
                train_acc.append(float(r.data.metrics["train_accuracy"]))
                val_acc.append(float(r.data.metrics["val_accuracy"]))
                epochs.append(float(r.data.metrics["best_epoch"]))
            parents = [
                r
                for r in runs
                if r.info.status == "FINISHED" and "outer0_test_accuracy" in r.data.metrics
            ]
            run = max(parents, key=lambda r: r.info.end_time or 0)
            test_acc.append(float(run.data.metrics["outer0_test_accuracy"]))
            oof_acc.append(float(run.data.metrics["oof_accuracy"]))
        t_mean = float(np.mean(train_acc))
        v_mean = float(np.mean(val_acc))
        rows.append(
            {
                "variant": variant,
                "mean_best_epoch": float(np.mean(epochs)),
                "max_best_epoch": float(np.max(epochs)),
                "train_accuracy": t_mean,
                "val_accuracy": v_mean,
                "gap_accuracy": t_mean - v_mean,
                "oof_accuracy": float(np.mean(oof_acc)),
                "test_accuracy": float(np.mean(test_acc)),
                "n_fold_runs": len(train_acc),
            }
        )
    # overall
    all_train = [r["train_accuracy"] for r in rows]
    # recompute overall from raw would be better; approximate mean of variant means
    client = MlflowClient(TRACKING_URI)
    train_all: List[float] = []
    val_all: List[float] = []
    test_all: List[float] = []
    oof_all: List[float] = []
    ep_all: List[float] = []
    for variant in VARIANTS:
        for task in TASKS:
            name = f"hcp_stage2_compare_{variant}_{task}_skipgcn"
            exp = client.get_experiment_by_name(name)
            runs = client.search_runs(exp.experiment_id, max_results=100)
            for r in runs:
                if r.info.status != "FINISHED":
                    continue
                if r.info.run_name.startswith("outer"):
                    train_all.append(float(r.data.metrics["train_accuracy"]))
                    val_all.append(float(r.data.metrics["val_accuracy"]))
                    ep_all.append(float(r.data.metrics["best_epoch"]))
                elif "outer0_test_accuracy" in r.data.metrics:
                    test_all.append(float(r.data.metrics["outer0_test_accuracy"]))
                    oof_all.append(float(r.data.metrics["oof_accuracy"]))
    t_m, v_m = float(np.mean(train_all)), float(np.mean(val_all))
    rows.append(
        {
            "variant": "ALL",
            "mean_best_epoch": float(np.mean(ep_all)),
            "max_best_epoch": float(np.max(ep_all)),
            "train_accuracy": t_m,
            "val_accuracy": v_m,
            "gap_accuracy": t_m - v_m,
            "oof_accuracy": float(np.mean(oof_all)),
            "test_accuracy": float(np.mean(test_all)),
            "n_fold_runs": len(train_all),
        }
    )
    return pd.DataFrame(rows)


def add_mean_row(df: pd.DataFrame, value_col: str = "accuracy") -> pd.DataFrame:
    """Добавить MEAN по задачам для каждого variant (и model, если есть)."""
    keys = [c for c in ("model", "variant") if c in df.columns]
    if "variant" not in df.columns:
        raise ValueError("нужна колонка variant")
    parts = [df]
    for _, group in df.groupby(keys, sort=False):
        mean_row = {c: group.iloc[0][c] if c in keys else (np.nan if c != "task" else "MEAN") for c in df.columns}
        mean_row["task"] = "MEAN"
        mean_row[value_col] = float(group[value_col].mean())
        if "roc_auc" in df.columns:
            mean_row["roc_auc"] = float(group["roc_auc"].mean())
        if "oof_accuracy" in df.columns:
            mean_row["oof_accuracy"] = float(group["oof_accuracy"].mean())
        if "n_test" in df.columns:
            mean_row["n_test"] = int(group["n_test"].sum())
        parts.append(pd.DataFrame([mean_row]))
    out = pd.concat(parts, ignore_index=True)
    # stable order
    task_order = list(TASKS) + ["MEAN"]
    out["task"] = pd.Categorical(out["task"], categories=task_order, ordered=True)
    if "variant" in out.columns:
        out["variant"] = pd.Categorical(out["variant"], categories=list(VARIANTS), ordered=True)
    sort_cols = [c for c in ("model", "variant", "task") if c in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True)


def pivot_acc(df: pd.DataFrame) -> pd.DataFrame:
    """Wide: task × variant accuracy."""
    wide = df.pivot(index="task", columns="variant", values="accuracy")
    return wide.reindex(index=list(TASKS) + ["MEAN"], columns=list(VARIANTS))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Top-n pooled (folds 0–3)")
    top_pooled = add_mean_row(top_n_accuracy_pooled(OUTER_FOLDS))
    top_pooled.to_csv(OUT_DIR / "top_n_accuracy_pooled.csv", index=False)
    pivot_acc(top_pooled).to_csv(OUT_DIR / "top_n_accuracy_pooled_wide.csv")

    log.info("Top-n outer 0")
    top0 = add_mean_row(top_n_accuracy_pooled((0,)))
    top0.to_csv(OUT_DIR / "top_n_accuracy_outer0.csv", index=False)
    pivot_acc(top0).to_csv(OUT_DIR / "top_n_accuracy_outer0_wide.csv")

    log.info("LogReg all pooled (folds 0–3), re-fit on feature cache")
    lr_pooled = add_mean_row(logreg_accuracy_fixed(OUTER_FOLDS, "all"))
    lr_pooled.to_csv(OUT_DIR / "logreg_all_accuracy_pooled.csv", index=False)
    pivot_acc(lr_pooled).to_csv(OUT_DIR / "logreg_all_accuracy_pooled_wide.csv")

    log.info("LogReg all outer 0")
    lr0 = add_mean_row(logreg_accuracy_fixed((0,), "all"))
    lr0.to_csv(OUT_DIR / "logreg_all_accuracy_outer0.csv", index=False)
    pivot_acc(lr0).to_csv(OUT_DIR / "logreg_all_accuracy_outer0_wide.csv")

    log.info("SkipGCN stage2 test accuracy")
    s2 = add_mean_row(mlflow_test_accuracy("hcp_stage2_compare", "skipgcn"))
    s2.to_csv(OUT_DIR / "skipgcn_stage2_accuracy.csv", index=False)
    pivot_acc(s2).to_csv(OUT_DIR / "skipgcn_stage2_accuracy_wide.csv")

    log.info("SkipGCN overfit accuracy table")
    overfit = skipgcn_overfit_accuracy()
    overfit.to_csv(OUT_DIR / "skipgcn_overfit_accuracy.csv", index=False)

    log.info("Smoke SkipGCN / CF-MSR")
    smoke_s = mlflow_test_accuracy("hcp_smoke_defaults", "skipgcn")
    smoke_c = mlflow_test_accuracy("hcp_smoke_defaults", "cfmsr")
    smoke = smoke_s.merge(
        smoke_c,
        on=["variant", "task"],
        suffixes=("_skipgcn", "_cfmsr"),
    )
    smoke["delta_acc"] = smoke["accuracy_cfmsr"] - smoke["accuracy_skipgcn"]
    # MEAN rows
    mean_rows = []
    for variant in VARIANTS:
        sub = smoke[smoke["variant"] == variant]
        mean_rows.append(
            {
                "variant": variant,
                "task": "MEAN",
                "accuracy_skipgcn": float(sub["accuracy_skipgcn"].mean()),
                "accuracy_cfmsr": float(sub["accuracy_cfmsr"].mean()),
                "roc_auc_skipgcn": float(sub["roc_auc_skipgcn"].mean()),
                "roc_auc_cfmsr": float(sub["roc_auc_cfmsr"].mean()),
                "delta_acc": float(sub["delta_acc"].mean()),
            }
        )
    smoke_full = pd.concat([smoke, pd.DataFrame(mean_rows)], ignore_index=True)
    smoke_full.to_csv(OUT_DIR / "smoke_accuracy.csv", index=False)

    log.info("Готово: %s", OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
