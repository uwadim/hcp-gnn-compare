"""
Cross-validation training.

Вложенная CV по ``splits.json``: outer folds × inner folds × seeds.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from bootstrap_metrics import bootstrap_roc_auc_ci, subject_from_stem
from config_utils import parse_optional_int, resolve_path
from cv_splits import select_outer_folds
from dataset import HCPDataModule
from mlflow_utils import (
    log_artifact_path,
    log_hparams,
    log_split_metrics,
    mlflow_run_context,
)
from model_factory import build_model
from train import (
    build_criteria,
    evaluate,
    fit_model,
    format_metrics,
    get_metric_names,
    get_metric_threshold,
    get_score_metric,
    log_metrics_block,
    normalize_metric_name,
    safe_metrics,
    set_seed,
    _is_mlflow_active,
)

log = logging.getLogger(__name__)


def _objective_metric_name(cv_cfg: DictConfig) -> str:
    """
    Objective metric name.

    Parameters
    ----------
    cv_cfg : DictConfig
        Ветка ``cv``.

    Returns
    -------
    str
        Имя метрики для Optuna (по умолчанию ``roc_auc``).
    """
    raw = cv_cfg.get("objective_metric", "roc_auc")
    if raw is None or str(raw).strip().lower() in {"", "none", "null"}:
        return "roc_auc"
    return normalize_metric_name(raw)


def _fold_objective_from_val(
    val_out: Dict[str, Any],
    fit_score: float,
    objective_metric: str,
) -> float:
    """
    Fold objective from validation.

    Parameters
    ----------
    val_out : Dict[str, Any]
        Результат ``evaluate`` на val.
    fit_score : float
        Score early stopping из ``fit_model``.
    objective_metric : str
        Метрика Optuna; ``loss`` → ``-val_loss``.

    Returns
    -------
    float
        Значение objective (больше — лучше).
    """
    if objective_metric in {"loss", "neg_loss"}:
        return -float(val_out.get("loss", float("nan")))
    if objective_metric in val_out:
        try:
            return float(val_out[objective_metric])
        except Exception:
            return float("nan")
    return float(fit_score)


def run_cv_training(cfg: DictConfig) -> float:
    """
    Run CV training.

    Для каждого outer fold и каждого inner fold обучает модель на
    complementary train, валидирует на inner held-out, опционально
    оценивает outer test (ансамбль).

    Parameters
    ----------
    cfg : DictConfig
        Полная Hydra-конфигурация.

    Returns
    -------
    float
        Среднее ``cv.objective_metric`` по fold/seed (для Optuna maximize).
        Early stopping использует ``training.monitor_metric`` (часто ``loss``).
    """
    data_cfg = cfg.data
    training_cfg = cfg.training
    cv_cfg = cfg.cv

    log_hparams(cfg)

    seed = int(training_cfg.seed)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metric_names = get_metric_names(training_cfg)
    metric_threshold = get_metric_threshold(training_cfg)
    score_metric = get_score_metric(training_cfg, metric_names)
    objective_metric = _objective_metric_name(cv_cfg)

    ensemble_seeds = int(cv_cfg.get("ensemble_seeds", 1))
    save_models = bool(cv_cfg.get("save_models", False))
    evaluate_test = bool(cv_cfg.get("evaluate_test", True))
    use_inner_folds = bool(cv_cfg.get("use_inner_folds", True))
    bootstrap_n = int(cv_cfg.get("bootstrap_n", 0) or 0)
    bootstrap_seed = int(cv_cfg.get("bootstrap_seed", seed))

    log.info(
        "CV: monitor=%s objective=%s evaluate_test=%s",
        training_cfg.get("monitor_metric", score_metric),
        objective_metric,
        evaluate_test,
    )

    dm = HCPDataModule(data_cfg=data_cfg, training_cfg=training_cfg)
    outer_fold = parse_optional_int(data_cfg.get("outer_fold", None))
    outer_folds = select_outer_folds(dm.splits, outer_fold)

    fold_monitor_scores: List[float] = []
    fold_objective_values: List[float] = []
    oof_records: List[Dict[str, Any]] = []
    test_prob_sum: Optional[np.ndarray] = None
    test_labels: Optional[np.ndarray] = None
    test_stems: Optional[List[str]] = None
    test_n = 0

    for outer in outer_folds:
        dm.setup_outer(outer)
        inner_ids = (
            [int(item["fold"]) for item in outer.inner]
            if use_inner_folds
            else [int(data_cfg.get("inner_val_fold", 0))]
        )

        for inner_id in inner_ids:
            for seed_i in range(ensemble_seeds):
                run_seed = seed + outer.fold * 1009 + inner_id * 97 + seed_i * 17
                set_seed(run_seed)

                nested_name = f"outer{outer.fold}_inner{inner_id}_seed{seed_i}"
                with mlflow_run_context(cfg, run_name=nested_name, nested=True):
                    dm.prepare_inner(inner_id)
                    train_loader = dm.train_loader()
                    val_loader = dm.val_loader()
                    test_loader = dm.test_loader()

                    assert dm.train_dataset is not None
                    assert dm.val_dataset is not None

                    log.info(
                        "CV outer=%d inner=%d seed=%d train=%d val=%d test=%d",
                        outer.fold,
                        inner_id,
                        seed_i,
                        len(dm.train_dataset),
                        len(dm.val_dataset),
                        len(dm.test_dataset) if dm.test_dataset is not None else 0,
                    )

                    model = build_model(
                        training_cfg=training_cfg,
                        num_nodes=dm.num_nodes,
                        input_dim=dm.node_input_dim,
                    ).to(device)

                    pos_weight = torch.tensor(
                        [dm.pos_weight],
                        dtype=torch.float,
                        device=device,
                    )
                    label_smoothing = float(
                        training_cfg.get("label_smoothing", 0.0) or 0.0
                    )
                    train_crit, eval_crit = build_criteria(
                        pos_weight=pos_weight,
                        label_smoothing=label_smoothing,
                    )

                    fit_result = fit_model(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        criterion=train_crit,
                        training_cfg=training_cfg,
                        device=device,
                        metric_names=metric_names,
                        metric_threshold=metric_threshold,
                        eval_criterion=eval_crit,
                    )
                    fold_monitor_scores.append(float(fit_result.best_score))

                    if _is_mlflow_active():
                        import mlflow

                        mlflow.log_metric("best_score", float(fit_result.best_score))
                        mlflow.log_metric("best_epoch", float(fit_result.best_epoch))

                    assert val_loader is not None
                    val_out = evaluate(
                        model=model,
                        loader=val_loader,
                        criterion=eval_crit,
                        device=device,
                        metric_names=metric_names,
                        metric_threshold=metric_threshold,
                    )
                    obj_value = _fold_objective_from_val(
                        val_out=val_out,
                        fit_score=fit_result.best_score,
                        objective_metric=objective_metric,
                    )
                    fold_objective_values.append(obj_value)
                    log_metrics_block(
                        f"outer{outer.fold}_inner{inner_id}_val",
                        val_out,
                        metric_names,
                    )
                    log_split_metrics(
                        f"outer{outer.fold}_inner{inner_id}_val",
                        val_out,
                        metric_names,
                    )
                    if _is_mlflow_active():
                        import mlflow

                        if not np.isnan(obj_value):
                            mlflow.log_metric("fold_objective", float(obj_value))

                    pred_df = val_out.get("predictions")
                    if isinstance(pred_df, pd.DataFrame):
                        for _, row in pred_df.iterrows():
                            local_idx = int(row["sample_idx"])
                            stem = (
                                dm.val_dataset.sample_ids[local_idx]
                                if 0 <= local_idx < len(dm.val_dataset.sample_ids)
                                else None
                            )
                            oof_records.append(
                                {
                                    "outer_fold": outer.fold,
                                    "inner_fold": inner_id,
                                    "seed": seed_i,
                                    "stem": stem,
                                    "subject": (
                                        dm.val_dataset.subjects[local_idx]
                                        if stem is not None
                                        else None
                                    ),
                                    "label": float(row["label"]),
                                    "prob": float(row["prob"]),
                                }
                            )

                    if evaluate_test and test_loader is not None:
                        test_out = evaluate(
                            model=model,
                            loader=test_loader,
                            criterion=eval_crit,
                            device=device,
                            metric_names=metric_names,
                            metric_threshold=metric_threshold,
                        )
                        tpred = test_out.get("predictions")
                        if isinstance(tpred, pd.DataFrame) and dm.test_dataset is not None:
                            probs = tpred["prob"].to_numpy(dtype=np.float64)
                            if test_prob_sum is None:
                                test_prob_sum = np.zeros_like(probs)
                                test_labels = dm.test_dataset.labels.copy()
                                test_stems = list(dm.test_dataset.sample_ids)
                            # Разные outer fold имеют разный test — накапливаем
                            # только внутри одного outer; при полном CV пишем
                            # отдельно по outer (см. ниже при смене outer).
                            if len(probs) == len(test_prob_sum):
                                test_prob_sum = test_prob_sum + probs
                                test_n += 1
                            else:
                                # Смена outer fold: сбрасываем аккумулятор.
                                _log_test_ensemble(
                                    cfg=cfg,
                                    training_cfg=training_cfg,
                                    metric_names=metric_names,
                                    metric_threshold=metric_threshold,
                                    prob_sum=test_prob_sum,
                                    n_models=test_n,
                                    labels=test_labels,
                                    stems=test_stems,
                                    outer_fold=None,
                                    bootstrap_n=bootstrap_n,
                                    bootstrap_seed=bootstrap_seed,
                                )
                                test_prob_sum = probs.copy()
                                test_labels = dm.test_dataset.labels.copy()
                                test_stems = list(dm.test_dataset.sample_ids)
                                test_n = 1

                    if save_models:
                        path = resolve_path(
                            f"cv_outer{outer.fold}_inner{inner_id}_seed{seed_i}.pt"
                        )
                        if path is None:
                            path = (
                                f"cv_outer{outer.fold}_inner{inner_id}_seed{seed_i}.pt"
                            )
                        torch.save(model.state_dict(), path)
                        log_artifact_path(cfg, path)

        # Финализируем test-ансамбль текущего outer.
        if test_prob_sum is not None and test_n > 0:
            _log_test_ensemble(
                cfg=cfg,
                training_cfg=training_cfg,
                metric_names=metric_names,
                metric_threshold=metric_threshold,
                prob_sum=test_prob_sum,
                n_models=test_n,
                labels=test_labels,
                stems=test_stems,
                outer_fold=outer.fold,
                bootstrap_n=bootstrap_n,
                bootstrap_seed=bootstrap_seed,
            )
            test_prob_sum = None
            test_n = 0

    oof_df = pd.DataFrame(oof_records)
    if not oof_df.empty:
        # Усреднение по seed для одного stem внутри outer/inner.
        grouped = (
            oof_df.groupby(
                ["outer_fold", "inner_fold", "stem", "subject", "label"],
                as_index=False,
            ).agg(oof_prob=("prob", "mean"))
        )
        oof_metrics = safe_metrics(
            y_true=grouped["label"].tolist(),
            y_score=grouped["oof_prob"].tolist(),
            metric_names=metric_names,
            threshold=metric_threshold,
        )
        log.info("OOF metrics: %s", format_metrics(oof_metrics, metric_names))
        log_split_metrics("oof", oof_metrics, metric_names)

        if bootstrap_n > 0:
            auc, lo, hi = bootstrap_roc_auc_ci(
                grouped["label"].to_numpy(),
                grouped["oof_prob"].to_numpy(),
                n_bootstrap=bootstrap_n,
                seed=bootstrap_seed,
                groups=grouped["subject"].astype(str).tolist(),
            )
            log.info("OOF ROC-AUC bootstrap 95%% CI: %.4f [%.4f, %.4f]", auc, lo, hi)
            if _is_mlflow_active():
                import mlflow

                mlflow.log_metric("oof_roc_auc", float(auc))
                mlflow.log_metric("oof_roc_auc_ci_low", float(lo))
                mlflow.log_metric("oof_roc_auc_ci_high", float(hi))

        oof_out = resolve_path(str(cv_cfg.get("oof_out", "cv_oof_predictions.csv")))
        if oof_out is None:
            oof_out = "cv_oof_predictions.csv"
        grouped.to_csv(oof_out, index=False)
        log_artifact_path(cfg, oof_out)

    monitor_arr = np.asarray(fold_monitor_scores, dtype=np.float64)
    objective_arr = np.asarray(fold_objective_values, dtype=np.float64)
    mean_monitor = float(np.nanmean(monitor_arr)) if monitor_arr.size else float("nan")
    std_monitor = float(np.nanstd(monitor_arr)) if monitor_arr.size else float("nan")
    mean_objective = (
        float(np.nanmean(objective_arr)) if objective_arr.size else float("nan")
    )
    std_objective = (
        float(np.nanstd(objective_arr)) if objective_arr.size else float("nan")
    )
    log.info(
        "CV monitor (%s): mean=%.4f std=%.4f n=%d",
        training_cfg.get("monitor_metric", score_metric),
        mean_monitor,
        std_monitor,
        len(fold_monitor_scores),
    )
    log.info(
        "CV objective (%s): mean=%.4f std=%.4f n=%d",
        objective_metric,
        mean_objective,
        std_objective,
        len(fold_objective_values),
    )

    if _is_mlflow_active():
        import mlflow

        mlflow.log_metric("cv_mean_monitor", mean_monitor)
        mlflow.log_metric("cv_std_monitor", std_monitor)
        mlflow.log_metric("cv_mean_objective", mean_objective)
        mlflow.log_metric("cv_std_objective", std_objective)
        mlflow.log_metric("cv_objective", mean_objective)

    return mean_objective


def _log_test_ensemble(
    cfg: DictConfig,
    training_cfg: DictConfig,
    metric_names: List[str],
    metric_threshold: float,
    prob_sum: Optional[np.ndarray],
    n_models: int,
    labels: Optional[np.ndarray],
    stems: Optional[List[str]],
    outer_fold: Optional[int],
    bootstrap_n: int = 0,
    bootstrap_seed: int = 42,
) -> None:
    if prob_sum is None or labels is None or stems is None or n_models <= 0:
        return
    probs = prob_sum / float(n_models)
    metrics = safe_metrics(
        y_true=labels.tolist(),
        y_score=probs.tolist(),
        metric_names=metric_names,
        threshold=metric_threshold,
    )
    tag = f"outer{outer_fold}_test" if outer_fold is not None else "test"
    log.info(
        "Test ensemble (%s, n_models=%d): %s",
        tag,
        n_models,
        format_metrics(metrics, metric_names),
    )
    log_split_metrics(tag, metrics, metric_names)
    subjects = [subject_from_stem(stem) for stem in stems]

    if bootstrap_n > 0:
        # Кластерный bootstrap: у субъекта несколько образцов (run × condition).
        auc, lo, hi = bootstrap_roc_auc_ci(
            labels,
            probs,
            n_bootstrap=bootstrap_n,
            seed=bootstrap_seed + (0 if outer_fold is None else int(outer_fold)),
            groups=subjects,
        )
        log.info(
            "Test ROC-AUC bootstrap 95%% CI (%s): %.4f [%.4f, %.4f]",
            tag,
            auc,
            lo,
            hi,
        )
        if _is_mlflow_active():
            import mlflow

            mlflow.log_metric(f"{tag}_roc_auc", float(auc))
            mlflow.log_metric(f"{tag}_roc_auc_ci_low", float(lo))
            mlflow.log_metric(f"{tag}_roc_auc_ci_high", float(hi))

    suffix = f"_outer{outer_fold}" if outer_fold is not None else ""
    pred_path = resolve_path(
        str(training_cfg.get("predictions_out", "test_predictions.csv")).replace(
            ".csv",
            f"{suffix}.csv",
        )
    )
    if pred_path is None:
        pred_path = f"test_predictions{suffix}.csv"
    pd.DataFrame(
        {
            "stem": stems,
            "subject": subjects,
            "prob": probs,
            "pred": (probs >= metric_threshold).astype(int),
            "label": labels,
        }
    ).to_csv(pred_path, index=False)
    log_artifact_path(cfg, pred_path)
