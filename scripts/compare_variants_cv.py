#!/usr/bin/env python3
"""
Сравнение вариантов данных по нескольким outer fold: ROC-AUC и 95% CI.

Берёт из MLflow test-предсказания ``main_compare.py`` / ``config_compare_cv`` для
каждой ячейки ``variant × task × outer fold`` и считает два согласованных
способа агрегации (каждый субъект лежит ровно в одном test-fold):

- ``pooled`` — предсказания всех fold объединяются (832 субъекта при 4 fold),
  один кластерный bootstrap по субъектам; основной способ;
- ``macro`` — AUC каждого fold считается отдельно (CI — bootstrap внутри fold),
  итог — среднее по fold; реплики среднего получаются независимым ресемплингом
  внутри каждого fold (стратифицированный bootstrap). Устойчив к тому, что
  вероятности из моделей разных fold плохо сопоставимы по калибровке.

Для обоих способов считаются парные ΔAUC между вариантами и поправки на
множественность (Romano–Wolf, Holm, BH) теми же функциями, что в
``compare_variants.py``. Дополнительно — AUC по каждому fold отдельно.

CI отражают только выборку test-субъектов; изменчивость обучения (общие train
между fold, seed) они не покрывают.

Результат: CSV + markdown в ``artifacts/variant_comparison_cv/{model}/``
и MLflow run в experiment ``hcp_variant_comparison_cv_{model}``.
"""

from __future__ import annotations

# 1. Standard library
import argparse
import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

# 2. External libraries
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# 3. Local
from batch_runner import (  # noqa: E402
    DEFAULT_TRACKING_URI,
    MODELS,
    OUTER_FOLDS,
    TASKS,
    VARIANTS,
    _mlflow_client,
    stage2_cv_experiment_name,
)
from bootstrap_metrics import subject_from_stem  # noqa: E402
from compare_variants import (  # noqa: E402
    DEFAULT_REFERENCE,
    TaskComparison,
    auc_table,
    compare_task,
    load_experiment_predictions,
    mean_over_tasks,
    pairwise_table,
    render_markdown,
)
from cv_splits import load_splits, validate_outer_folds  # noqa: E402

log = logging.getLogger("compare_variants_cv")

DEFAULT_SPLITS = (
    ROOT / ".." / "HPC_data_ref" / "hcp_mmp_mean_ablation_graphs" / "splits.json"
)

SKIPGCN_NOTE = "HP SkipGCN зафиксированы (`conf/best_task`, подобраны на outer 0)"

# Предсказания variant -> DataFrame(stem, prob, label) одного fold.
FoldPredictions = Mapping[str, pd.DataFrame]


@dataclass(frozen=True)
class CvTaskComparison:
    """
    Результат сравнения вариантов на одной задаче по нескольким fold.

    Attributes
    ----------
    task : str
        Задача HCP.
    pooled : TaskComparison
        AUC на объединённых предсказаниях всех fold.
    per_fold : Dict[int, TaskComparison]
        AUC отдельно по каждому fold.
    macro : TaskComparison
        Среднее AUC по fold со стратифицированными репликами.
    """

    task: str
    pooled: TaskComparison
    per_fold: Dict[int, TaskComparison]
    macro: TaskComparison


def normalize_folds(folds: Sequence[int]) -> List[int]:
    """
    Validate and sort a list of outer folds.

    Parameters
    ----------
    folds : Sequence[int]
        Номера outer fold.

    Returns
    -------
    List[int]
        Отсортированные уникальные fold.

    Raises
    ------
    ValueError
        Если список пуст или содержит fold вне ``OUTER_FOLDS``.
    """
    fold_list = sorted({int(k) for k in folds})
    if not fold_list or set(fold_list) - set(OUTER_FOLDS):
        raise ValueError(f"folds должен быть непустым подмножеством {OUTER_FOLDS}, получено {list(folds)}")
    return fold_list


def expected_test_subjects(splits_path: Path, folds: Sequence[int]) -> Dict[int, Set[str]]:
    """
    Read expected test subjects of outer folds from ``splits.json``.

    Parameters
    ----------
    splits_path : Path
        Путь к ``splits.json``.
    folds : Sequence[int]
        Нужные outer fold.

    Returns
    -------
    Dict[int, Set[str]]
        Test-субъекты по fold.

    Raises
    ------
    ValueError
        Если fold нет в ``splits.json`` или test-множества пересекаются.
    """
    outer = {split.fold: split for split in load_splits(splits_path)}
    validate_outer_folds(list(outer.values()))
    missing = sorted(set(folds) - set(outer))
    if missing:
        raise ValueError(f"В {splits_path} нет outer fold {missing}")
    expected = {k: set(outer[k].test_subjects) for k in folds}
    for i, first in enumerate(folds):
        for second in folds[i + 1:]:
            overlap = expected[first] & expected[second]
            if overlap:
                raise ValueError(
                    f"test fold {first} и {second} пересекаются: {sorted(overlap)[:5]}"
                )
    return expected


def validate_fold_subjects(
    preds_by_fold: Mapping[int, FoldPredictions],
    expected: Mapping[int, Set[str]],
) -> None:
    """
    Check that predictions of every fold cover exactly its test subjects.

    Parameters
    ----------
    preds_by_fold : Mapping[int, FoldPredictions]
        Предсказания ``fold -> variant -> DataFrame``.
    expected : Mapping[int, Set[str]]
        Ожидаемые test-субъекты по fold.

    Raises
    ------
    ValueError
        Если набор субъектов fold не совпадает с ожидаемым.
    """
    for fold, by_variant in preds_by_fold.items():
        if fold not in expected:
            raise ValueError(f"fold {fold}: нет ожидаемого набора субъектов")
        for variant, frame in by_variant.items():
            got = set(frame["stem"].map(subject_from_stem))
            if got != expected[fold]:
                raise ValueError(
                    f"fold {fold}, {variant}: субъекты предсказаний не совпадают со splits.json "
                    f"(лишних {len(got - expected[fold])}, не хватает {len(expected[fold] - got)})"
                )


def pool_predictions(preds_by_fold: Mapping[int, FoldPredictions]) -> Dict[str, pd.DataFrame]:
    """
    Concatenate predictions of all folds per variant.

    Parameters
    ----------
    preds_by_fold : Mapping[int, FoldPredictions]
        Предсказания ``fold -> variant -> DataFrame``.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Объединённые предсказания по вариантам (stem уникальны между fold).

    Raises
    ------
    ValueError
        Если пусто или набор вариантов различается между fold.
    """
    if not preds_by_fold:
        raise ValueError("preds_by_fold пуст")
    folds = sorted(preds_by_fold)
    variants = list(preds_by_fold[folds[0]])
    if any(list(preds_by_fold[k]) != variants for k in folds):
        raise ValueError("Набор вариантов различается между fold")
    return {
        variant: pd.concat([preds_by_fold[k][variant] for k in folds], ignore_index=True)
        for variant in variants
    }


def macro_over_folds(task: str, per_fold: Mapping[int, TaskComparison]) -> TaskComparison:
    """
    Average AUC over folds with independent per-fold replicates.

    Реплики среднего — среднее реплик fold (в каждом fold ресемплируются свои
    субъекты независимо) — это стратифицированный по fold bootstrap.

    Parameters
    ----------
    task : str
        Задача HCP.
    per_fold : Mapping[int, TaskComparison]
        Сравнения по fold с одинаковым набором вариантов и числом реплик.

    Returns
    -------
    TaskComparison
        Среднее AUC по fold и его реплики.
    """
    if not per_fold:
        raise ValueError("per_fold пуст")
    results = [per_fold[k] for k in sorted(per_fold)]
    variants = list(results[0].point)
    if any(list(r.point) != variants for r in results):
        raise ValueError("Набор вариантов различается между fold")
    point = {v: float(np.mean([r.point[v] for r in results])) for v in variants}
    reps = {v: np.mean(np.vstack([r.replicates[v] for r in results]), axis=0) for v in variants}
    subjects = tuple(sorted(set().union(*(r.subjects for r in results))))
    return TaskComparison(task=task, point=point, replicates=reps, subjects=subjects)


def compare_task_cv(
    task: str,
    preds_by_fold: Mapping[int, FoldPredictions],
    n_bootstrap: int,
    seed: int,
) -> CvTaskComparison:
    """
    Compare variants on one task over several folds.

    Parameters
    ----------
    task : str
        Задача HCP.
    preds_by_fold : Mapping[int, FoldPredictions]
        Предсказания ``fold -> variant -> DataFrame``.
    n_bootstrap : int
        Число bootstrap-реплик.
    seed : int
        Random seed; fold ``k`` использует ``seed + k`` (одинаково для всех задач,
        поэтому реплики общие между задачами и вариантами).

    Returns
    -------
    CvTaskComparison
        AUC pooled, по fold и macro.
    """
    pooled = compare_task(task, pool_predictions(preds_by_fold), n_bootstrap, seed)
    per_fold = {
        fold: compare_task(task, dict(preds), n_bootstrap, seed + fold)
        for fold, preds in sorted(preds_by_fold.items())
    }
    return CvTaskComparison(
        task=task,
        pooled=pooled,
        per_fold=per_fold,
        macro=macro_over_folds(task, per_fold),
    )


def summarize_tasks(results: Sequence[CvTaskComparison]) -> Tuple[
    List[TaskComparison],
    List[TaskComparison],
    Dict[int, List[TaskComparison]],
]:
    """
    Build per-aggregation lists of task results with the MEAN row.

    Parameters
    ----------
    results : Sequence[CvTaskComparison]
        Результаты по задачам.

    Returns
    -------
    Tuple[List[TaskComparison], List[TaskComparison], Dict[int, List[TaskComparison]]]
        Списки pooled и macro (с ``MEAN`` при ≥2 задачах) и per-fold (с ``MEAN``).
    """
    if not results:
        raise ValueError("results пуст")
    pooled = [r.pooled for r in results]
    macro = [r.macro for r in results]
    folds = sorted(results[0].per_fold)
    per_fold = {k: [r.per_fold[k] for r in results] for k in folds}
    if len(results) > 1:
        pooled.append(mean_over_tasks(pooled))
        macro.append(mean_over_tasks(macro))
        for k in folds:
            per_fold[k].append(mean_over_tasks(per_fold[k]))
    return pooled, macro, per_fold


def fold_auc_table(per_fold: Mapping[int, Sequence[TaskComparison]], alpha: float = 0.05) -> pd.DataFrame:
    """
    AUC table by fold with per-fold CI.

    Parameters
    ----------
    per_fold : Mapping[int, Sequence[TaskComparison]]
        Результаты по задачам для каждого fold.
    alpha : float, optional
        Уровень CI.

    Returns
    -------
    pd.DataFrame
        ``fold, task, variant, roc_auc, ci_low, ci_high, rank``.
    """
    frames = []
    for fold, results in sorted(per_fold.items()):
        table = auc_table(list(results), alpha=alpha)
        table.insert(0, "fold", fold)
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


def render_fold_markdown(fold_table: pd.DataFrame) -> str:
    """
    Render per-fold AUC table.

    Parameters
    ----------
    fold_table : pd.DataFrame
        Таблица :func:`fold_auc_table`.

    Returns
    -------
    str
        Markdown: AUC каждого fold, среднее и SD между fold.
    """
    folds = sorted(fold_table["fold"].unique())
    lines = [
        "## AUC по outer fold",
        "",
        "ROC-AUC на test каждого fold; 95% CI по fold — в `auc_per_fold.csv`. "
        f"SD — выборочное (ddof=1) между fold; описательная статистика при n={len(folds)}, не интервал.",
        "",
        "| task | variant | " + " | ".join(f"fold {k}" for k in folds) + " | mean | SD |",
        "|---|---|" + "---|" * (len(folds) + 2),
    ]
    order = list(dict.fromkeys(zip(fold_table["task"], fold_table["variant"])))
    for task, variant in order:
        sub = fold_table[(fold_table["task"] == task) & (fold_table["variant"] == variant)]
        values = sub.set_index("fold")["roc_auc"].reindex(folds).to_numpy(dtype=float)
        sd = float(np.std(values, ddof=1)) if values.size > 1 else float("nan")
        cells = " | ".join(f"{v:.4f}" for v in values)
        lines.append(f"| {task} | {variant} | {cells} | {np.mean(values):.4f} | {sd:.4f} |")
    return "\n".join(lines) + "\n"


def render_cv_report(
    model: str,
    folds: Sequence[int],
    n_subjects: int,
    n_bootstrap: int,
    pooled_md: str,
    macro_md: str,
    fold_md: str,
    note: str = SKIPGCN_NOTE,
) -> str:
    """
    Assemble the final markdown report.

    Parameters
    ----------
    model : str
        Модель.
    folds : Sequence[int]
        Использованные outer fold.
    n_subjects : int
        Число субъектов в pooled.
    n_bootstrap : int
        Число bootstrap-реплик.
    pooled_md : str
        Отчёт pooled (:func:`render_markdown`).
    macro_md : str
        Отчёт macro.
    fold_md : str
        Таблица по fold.
    note : str, optional
        Строка об условиях обучения модели (маркированный пункт отчёта).

    Returns
    -------
    str
        Markdown-отчёт.
    """
    header = [
        f"# Сравнение вариантов по outer fold: {model}",
        "",
        f"- outer fold: {', '.join(str(k) for k in folds)}; субъектов в pooled: {n_subjects} "
        "(каждый — в test ровно одного fold)",
        f"- {note}; bootstrap: кластерный по субъектам, B={n_bootstrap}",
        "- pooled — основной результат; macro — проверка устойчивости. Вывод считается "
        "устойчивым, если ΔAUC значим (p_RW < α) в обоих и знак совпадает.",
        "- CI покрывают выборку test-субъектов, но не изменчивость обучения "
        "(общие train у fold, seed).",
        "",
    ]
    # Заголовки вложенных отчётов опускаются на уровень ниже.
    nested = [
        "\n".join(("#" + line if line.startswith("#") else line) for line in md.splitlines())
        for md in (pooled_md, macro_md)
    ]
    return "\n".join(header) + "\n" + nested[0] + "\n\n" + nested[1] + "\n\n" + fold_md


def write_cv_comparison(
    results: Sequence[CvTaskComparison],
    label: str,
    folds: Sequence[int],
    out_dir: Path,
    n_bootstrap: int,
    alpha: float = 0.05,
    reference: Optional[str] = DEFAULT_REFERENCE,
    note: str = SKIPGCN_NOTE,
    quiet: bool = False,
) -> Tuple[Dict[str, Tuple[pd.DataFrame, pd.DataFrame]], int]:
    """
    Build and write the multi-fold comparison report of one model.

    Пишет ``report.md``, ``auc_{pooled,macro}.csv``, ``pairwise_{pooled,macro}.csv``
    и ``auc_per_fold.csv`` в ``out_dir``. Общая часть SkipGCN и baseline.

    Parameters
    ----------
    results : Sequence[CvTaskComparison]
        Результаты по задачам.
    label : str
        Название модели / классификатора для заголовков.
    folds : Sequence[int]
        Использованные outer fold.
    out_dir : Path
        Каталог отчёта.
    n_bootstrap : int
        Число реплик (для подписи).
    alpha : float, optional
        Уровень CI и значимости.
    reference : Optional[str], optional
        Эталон основного семейства; ``None`` — одно семейство.
    note : str, optional
        Условия обучения модели для шапки отчёта.
    quiet : bool, optional
        Не печатать отчёт в stdout.

    Returns
    -------
    Tuple[Dict[str, Tuple[pd.DataFrame, pd.DataFrame]], int]
        ``{"pooled"|"macro": (таблица AUC, таблица парных ΔAUC)}`` и число
        субъектов в pooled.
    """
    pooled_res, macro_res, fold_res = summarize_tasks(results)
    tables = {}
    for name, res in (("pooled", pooled_res), ("macro", macro_res)):
        tables[name] = (auc_table(res, alpha=alpha), pairwise_table(res, reference=reference, alpha=alpha))
    fold_table = fold_auc_table(fold_res, alpha=alpha)

    n_subjects = len(results[0].pooled.subjects)
    scope = f"outer fold {', '.join(str(k) for k in folds)}"
    mds = {
        name: render_markdown(
            label,
            auc,
            pairs,
            n_bootstrap,
            alpha=alpha,
            title=f"{'Pooled' if name == 'pooled' else 'Macro (среднее по fold)'}: {label}, {scope}",
        )
        for name, (auc, pairs) in tables.items()
    }
    report = render_cv_report(
        label, folds, n_subjects, n_bootstrap,
        mds["pooled"], mds["macro"], render_fold_markdown(fold_table), note=note,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, (auc, pairs) in tables.items():
        auc.to_csv(out_dir / f"auc_{name}.csv", index=False)
        pairs.to_csv(out_dir / f"pairwise_{name}.csv", index=False)
    fold_table.to_csv(out_dir / "auc_per_fold.csv", index=False)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    if not quiet:
        print(report)
    log.info("Отчёт записан в %s", out_dir)
    return tables, n_subjects


def load_cv_predictions(
    client,
    model: str,
    task: str,
    folds: Sequence[int],
    variants: Sequence[str],
    tmp_dir: Path,
) -> Tuple[Dict[int, Dict[str, pd.DataFrame]], Dict[str, str]]:
    """
    Load test predictions of one task for all folds and variants.

    Parameters
    ----------
    client : MlflowClient
        MLflow client.
    model : str
        Модель.
    task : str
        Задача.
    folds : Sequence[int]
        Outer fold.
    variants : Sequence[str]
        Варианты данных.
    tmp_dir : Path
        Каталог для скачивания артефактов.

    Returns
    -------
    Tuple[Dict[int, Dict[str, pd.DataFrame]], Dict[str, str]]
        Предсказания ``fold -> variant -> DataFrame`` (только найденные) и
        run_id по ключу ``outer{K}_{variant}``.
    """
    preds: Dict[int, Dict[str, pd.DataFrame]] = {}
    run_ids: Dict[str, str] = {}
    for fold in folds:
        for variant in variants:
            loaded = load_experiment_predictions(
                client,
                stage2_cv_experiment_name(variant, task, model, fold),
                f"test_predictions_outer{fold}.csv",
                tmp_dir / f"{variant}_{task}_outer{fold}",
            )
            if loaded is None:
                continue
            preds.setdefault(fold, {})[variant] = loaded[0]
            run_ids[f"outer{fold}_{variant}"] = loaded[1]
    return preds, run_ids


def _is_complete(preds: Mapping[int, FoldPredictions], folds: Sequence[int], variants: Sequence[str]) -> bool:
    return all(len(preds.get(k, {})) == len(variants) for k in folds)


def _log_to_mlflow(
    tracking_uri: str,
    model: str,
    folds: Sequence[int],
    pooled: pd.DataFrame,
    macro: pd.DataFrame,
    out_dir: Path,
    params: Dict[str, object],
    run_ids: Dict[str, Dict[str, str]],
) -> None:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(f"hcp_variant_comparison_cv_{model}")
    folds_tag = "".join(str(k) for k in folds)
    with mlflow.start_run(run_name=f"variants_cv_{model}_folds{folds_tag}"):
        mlflow.set_tags({"model": model, "stage": "stage2_cv", "outer_folds": folds_tag})
        mlflow.log_params({**params, "bootstrap": "cluster_subject", "fwer": "romano_wolf"})
        for task, ids in run_ids.items():
            for key, run_id in ids.items():
                mlflow.set_tag(f"source_{task}_{key}", run_id)
        for agg, table in (("pooled", pooled), ("macro", macro)):
            for row in table.itertuples(index=False):
                prefix = f"{agg}_{row.task}_{row.variant}"
                mlflow.log_metrics(
                    {
                        f"{prefix}_roc_auc": float(row.roc_auc),
                        f"{prefix}_ci_low": float(row.ci_low),
                        f"{prefix}_ci_high": float(row.ci_high),
                    }
                )
        mlflow.log_artifacts(str(out_dir))


def run_compare_variants_cv(
    model: str,
    folds: Sequence[int],
    tracking_uri: str,
    splits_path: Path,
    out_dir: Path,
    n_bootstrap: int = 10000,
    seed: int = 42,
    reference: Optional[str] = DEFAULT_REFERENCE,
    alpha: float = 0.05,
    log_mlflow: bool = True,
) -> int:
    """
    Run multi-fold variant comparison.

    Parameters
    ----------
    model : str
        Модель.
    folds : Sequence[int]
        Outer fold для агрегации.
    tracking_uri : str
        MLflow tracking URI.
    splits_path : Path
        ``splits.json`` архива (проверка состава test-субъектов).
    out_dir : Path
        Каталог отчёта.
    n_bootstrap : int, optional
        Число реплик.
    seed : int, optional
        Random seed.
    reference : Optional[str], optional
        Эталон основного семейства; ``None`` — одно семейство.
    alpha : float, optional
        Уровень значимости.
    log_mlflow : bool, optional
        Писать ли отдельный MLflow run.

    Returns
    -------
    int
        0 — успех, 1 — нет ни одной задачи со всеми fold и вариантами.
    """
    if model not in MODELS:
        raise ValueError(f"model должен быть одним из {MODELS}, получено {model!r}")
    fold_list = normalize_folds(folds)
    if n_bootstrap < 100:
        raise ValueError(f"n_bootstrap слишком мал: {n_bootstrap}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha должен быть в (0, 1), получено {alpha}")
    if reference is not None and reference not in VARIANTS:
        raise ValueError(f"reference должен быть одним из {VARIANTS}, получено {reference!r}")

    expected = expected_test_subjects(splits_path, fold_list)
    client = _mlflow_client(tracking_uri)
    results: List[CvTaskComparison] = []
    run_ids: Dict[str, Dict[str, str]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for task in TASKS:
            preds, ids = load_cv_predictions(client, model, task, fold_list, VARIANTS, Path(tmp))
            if not _is_complete(preds, fold_list, VARIANTS):
                have = sorted(ids)
                log.warning("Задача %s пропущена: не все fold × variant (найдено %d из %d)",
                            task, len(have), len(fold_list) * len(VARIANTS))
                continue
            validate_fold_subjects(preds, expected)
            results.append(compare_task_cv(task, preds, n_bootstrap=n_bootstrap, seed=seed))
            run_ids[task] = ids

    if not results:
        log.error("Нет ни одной задачи со всеми fold %s и %d вариантами", fold_list, len(VARIANTS))
        return 1

    tables, n_subjects = write_cv_comparison(
        results,
        label=model,
        folds=fold_list,
        out_dir=out_dir,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        reference=reference,
    )

    if log_mlflow:
        params: Dict[str, object] = {
            "n_bootstrap": n_bootstrap,
            "seed": seed,
            "reference": str(reference),
            "alpha": alpha,
            "n_subjects_pooled": n_subjects,
        }
        _log_to_mlflow(
            tracking_uri, model, fold_list,
            tables["pooled"][0], tables["macro"][0], out_dir, params, run_ids,
        )
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    Parse CLI arguments.

    Parameters
    ----------
    argv : Optional[Sequence[str]]
        Аргументы CLI.

    Returns
    -------
    argparse.Namespace
        Разобранные аргументы.
    """
    parser = argparse.ArgumentParser(description="Сравнение вариантов данных по нескольким outer fold")
    parser.add_argument("--model", choices=MODELS, default="skipgcn")
    parser.add_argument("--folds", type=int, nargs="+", default=list(OUTER_FOLDS), metavar="K")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reference",
        default=DEFAULT_REFERENCE,
        help="Эталон основного семейства гипотез; 'none' — все пары в одном семействе",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--no-mlflow", action="store_true", help="Не создавать MLflow run")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Main.

    Parameters
    ----------
    argv : Optional[Sequence[str]]
        Аргументы CLI.

    Returns
    -------
    int
        Код выхода.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args(argv)
    out_dir = args.out_dir or ROOT / "artifacts" / "variant_comparison_cv" / args.model
    return run_compare_variants_cv(
        model=args.model,
        folds=args.folds,
        tracking_uri=args.tracking_uri,
        splits_path=Path(args.splits).resolve(),
        out_dir=Path(out_dir).resolve(),
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        reference=None if args.reference.lower() == "none" else args.reference,
        alpha=args.alpha,
        log_mlflow=not args.no_mlflow,
    )


if __name__ == "__main__":
    raise SystemExit(main())
