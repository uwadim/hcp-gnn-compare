#!/usr/bin/env python3
"""
Классификатор «средний вес ребра» и сравнение вариантов данных.

Скор графа — среднее по верхнему треугольнику матрицы весов (без порога и
без обучения GNN). Знак скора выбирается по train pool (AUC ≥ 0.5 → +1),
затем скор применяется к outer test. Для всех 4 вариантов × 7 задач:
ROC-AUC [95% CI] кластерным bootstrap по субъектам, парные ΔAUC и поправки
на множественность — через те же функции, что в ``compare_variants.py``.

Важно: веса ``correlation``-графов стандартизованы по верхнему треугольнику,
поэтому их средний вес тождественно равен нулю (шум float, ~1e-17), и такой
классификатор вырожден. Скрипт это определяет и помечает в отчёте.

Результат: ``artifacts/mean_edge_baseline/`` (``report.md`` — один отчёт по всем outer
fold: pooled и macro AUC с CI, AUC по fold; ``auc_{pooled,macro}.csv``,
``pairwise_{pooled,macro}.csv``, ``auc_per_fold.csv``, ``cell_diagnostics.csv``).
CI считаются по всем fold, см. ``compare_variants_cv.py``.
"""

from __future__ import annotations

# 1. Standard library
import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# 2. External libraries
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# 3. Local
from batch_runner import OUTER_FOLDS, TASKS, VARIANTS  # noqa: E402
from compare_variants import DEFAULT_REFERENCE  # noqa: E402
from compare_variants_cv import (  # noqa: E402
    CvTaskComparison,
    compare_task_cv,
    expected_test_subjects,
    normalize_folds,
    validate_fold_subjects,
    write_cv_comparison,
)
from config_utils import resolve_existing_path  # noqa: E402
from cv_splits import select_outer_folds  # noqa: E402
from dataset import GraphSample, HCPDataModule  # noqa: E402

log = logging.getLogger("mean_edge_baseline")

MODEL_NAME = "mean_edge_baseline"
# Размах скора ниже порога — скор константа (шум float), AUC не имеет смысла.
DEGENERATE_PTP = 1e-9
CACHE_COLUMNS = ("stem", "subject", "label", "role", "score")


@dataclass(frozen=True)
class CellResult:
    """
    Результат классификатора для одной ячейки variant × task.

    Attributes
    ----------
    variant : str
        Вариант данных.
    task : str
        Задача HCP.
    sign : int
        Знак скора (+1 или −1), выбранный по train pool.
    train_auc : float
        AUC скора со знаком на train pool.
    degenerate : bool
        Скор практически константа.
    test : pd.DataFrame
        Колонки ``stem``, ``prob`` (скор со знаком), ``label``.
    outer_fold : int
        Outer fold ячейки.
    """

    variant: str
    task: str
    sign: int
    train_auc: float
    degenerate: bool
    test: pd.DataFrame
    outer_fold: int = 0


@lru_cache(maxsize=4)
def _upper_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(n, k=1)


def mean_edge_weight(matrix: np.ndarray) -> float:
    """
    Mean edge weight.

    Parameters
    ----------
    matrix : np.ndarray
        Квадратная матрица весов ``(N, N)``.

    Returns
    -------
    float
        Среднее по верхнему треугольнику без диагонали.

    Raises
    ------
    ValueError
        Если матрица не квадратная или имеет размер < 2.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError(f"Ожидалась квадратная матрица N≥2, shape={matrix.shape}")
    return float(matrix[_upper_indices(matrix.shape[0])].mean(dtype=np.float64))


def _score_file(path: Path) -> float:
    return mean_edge_weight(np.load(path))


def score_samples(samples: Sequence[GraphSample], workers: int) -> np.ndarray:
    """
    Score samples.

    Parameters
    ----------
    samples : Sequence[GraphSample]
        Образцы с путями к матрицам рёбер.
    workers : int
        Число потоков чтения.

    Returns
    -------
    np.ndarray
        Скоры в порядке ``samples``.
    """
    if workers < 1:
        raise ValueError(f"workers должен быть ≥ 1, получено {workers}")
    paths = [s.edge_path for s in samples]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return np.asarray(list(pool.map(_score_file, paths)), dtype=np.float64)


def fit_sign(labels: np.ndarray, scores: np.ndarray) -> tuple[int, float]:
    """
    Fit score sign on train.

    Parameters
    ----------
    labels : np.ndarray
        Бинарные метки train.
    scores : np.ndarray
        Скоры train.

    Returns
    -------
    tuple[int, float]
        ``(sign, auc_signed)``; для константного скора ``(+1, nan)``.
    """
    from bootstrap_metrics import _rank_auc

    y = labels.astype(int)
    if len(np.unique(y)) < 2:
        raise ValueError("В train один класс — знак скора не определён")
    if is_degenerate(scores):
        return 1, float("nan")
    auc = _rank_auc(y, scores)
    return (1, auc) if auc >= 0.5 else (-1, 1.0 - auc)


def is_degenerate(scores: np.ndarray) -> bool:
    """
    Is degenerate.

    Parameters
    ----------
    scores : np.ndarray
        Скоры.

    Returns
    -------
    bool
        ``True``, если размах скоров ниже порога.
    """
    return bool(np.ptp(scores) < DEGENERATE_PTP)


def load_samples(
    data_root: Path,
    variant: str,
    task: str,
    outer_fold: int,
) -> tuple[List[GraphSample], List[GraphSample]]:
    """
    Load train pool and test samples of one outer fold.

    Parameters
    ----------
    data_root : Path
        Корень архива HCP.
    variant : str
        Вариант данных.
    task : str
        Задача HCP.
    outer_fold : int
        Outer fold.

    Returns
    -------
    tuple[List[GraphSample], List[GraphSample]]
        ``(train_pool, test)``.
    """
    cfg = OmegaConf.create(
        {
            "root": str(data_root),
            "variant": variant,
            "task": task,
            "node_features": "ones",
            "outer_fold": outer_fold,
        }
    )
    dm = HCPDataModule(cfg)
    dm.setup_outer(select_outer_folds(dm.splits, outer_fold)[0])
    return dm.pool_samples, dm.test_samples


def _cache_path(cache_dir: Path, variant: str, task: str, outer_fold: int) -> Path:
    return cache_dir / f"{variant}_{task}_outer{outer_fold}.csv"


def collect_scores(
    data_root: Path,
    variant: str,
    task: str,
    outer_fold: int,
    workers: int,
    cache_dir: Optional[Path],
) -> pd.DataFrame:
    """
    Collect mean-edge scores of train pool and test.

    Parameters
    ----------
    data_root : Path
        Корень архива HCP.
    variant : str
        Вариант данных.
    task : str
        Задача HCP.
    outer_fold : int
        Outer fold.
    workers : int
        Число потоков чтения.
    cache_dir : Optional[Path]
        Каталог кэша скоров; ``None`` — без кэша.

    Returns
    -------
    pd.DataFrame
        Колонки ``stem, subject, label, role, score``; ``role`` — train|test.
    """
    cache = None if cache_dir is None else _cache_path(cache_dir, variant, task, outer_fold)
    if cache is not None and cache.is_file():
        frame = pd.read_csv(cache, dtype={"stem": str, "subject": str})
        if list(frame.columns) == list(CACHE_COLUMNS):
            return frame
        log.warning("Кэш %s имеет другие колонки — пересчёт", cache)

    pool_samples, test_samples = load_samples(data_root, variant, task, outer_fold)
    parts = []
    for role, samples in (("train", pool_samples), ("test", test_samples)):
        parts.append(
            pd.DataFrame(
                {
                    "stem": [s.stem for s in samples],
                    "subject": [s.subject for s in samples],
                    "label": [float(s.label) for s in samples],
                    "role": role,
                    "score": score_samples(samples, workers),
                }
            )
        )
    frame = pd.concat(parts, ignore_index=True)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cache, index=False)
    return frame


def build_cell(frame: pd.DataFrame, variant: str, task: str) -> CellResult:
    """
    Build cell result from scores.

    Parameters
    ----------
    frame : pd.DataFrame
        Результат :func:`collect_scores`.
    variant : str
        Вариант данных.
    task : str
        Задача HCP.

    Returns
    -------
    CellResult
        Знак по train и предсказания test.
    """
    train = frame[frame["role"] == "train"]
    test = frame[frame["role"] == "test"]
    if train.empty or test.empty:
        raise ValueError(f"{variant}/{task}: пустой train или test")
    sign, train_auc = fit_sign(train["label"].to_numpy(), train["score"].to_numpy())
    return CellResult(
        variant=variant,
        task=task,
        sign=sign,
        train_auc=train_auc,
        degenerate=is_degenerate(frame["score"].to_numpy()),
        test=pd.DataFrame(
            {
                "stem": test["stem"].to_numpy(),
                "prob": sign * test["score"].to_numpy(),
                "label": test["label"].to_numpy(),
            }
        ),
    )


def diagnostics_table(cells: Sequence[CellResult]) -> pd.DataFrame:
    """
    Diagnostics table.

    Parameters
    ----------
    cells : Sequence[CellResult]
        Результаты ячеек.

    Returns
    -------
    pd.DataFrame
        ``fold, task, variant, sign, train_auc, score_degenerate``.
    """
    return pd.DataFrame(
        [
            {
                "fold": c.outer_fold,
                "task": c.task,
                "variant": c.variant,
                "sign": c.sign,
                "train_auc": c.train_auc,
                "score_degenerate": c.degenerate,
            }
            for c in cells
        ]
    )


def render_diagnostics(diag: pd.DataFrame) -> str:
    """
    Render diagnostics markdown.

    Parameters
    ----------
    diag : pd.DataFrame
        Таблица :func:`diagnostics_table`.

    Returns
    -------
    str
        Markdown-раздел.
    """
    lines = [
        "",
        "## Классификатор и диагностика",
        "",
        "Скор — среднее по верхнему треугольнику матрицы весов; знак выбран по train pool "
        "(AUC ≥ 0.5 → +1). Без обучения GNN.",
    ]
    bad = sorted(set(diag.loc[diag["score_degenerate"], "variant"]))
    if bad:
        lines += [
            "",
            f"**Вырожденный скор у: {', '.join(bad)}.** Размах среднего веса < {DEGENERATE_PTP:g} "
            "(веса стандартизованы по верхнему треугольнику, среднее ≡ 0), AUC ≈ 0.5 — шум; "
            "сравнение с этими вариантами показывает лишь, что другие варианты лучше шума.",
        ]
    lines += [
        "",
        "| fold | task | variant | знак | AUC на train | вырожден |",
        "|---|---|---|---|---|---|",
    ]
    for row in diag.itertuples(index=False):
        auc = "—" if np.isnan(row.train_auc) else f"{row.train_auc:.4f}"
        lines.append(
            f"| {row.fold} | {row.task} | {row.variant} | {'+' if row.sign > 0 else '−'} | {auc} | "
            f"{'да' if row.score_degenerate else 'нет'} |"
        )
    return "\n".join(lines) + "\n"


def run_baseline(
    data_root: Path,
    out_dir: Path,
    tasks: Sequence[str],
    outer_folds: Sequence[int] = OUTER_FOLDS,
    n_bootstrap: int = 10000,
    seed: int = 42,
    reference: Optional[str] = DEFAULT_REFERENCE,
    alpha: float = 0.05,
    workers: int = 8,
    use_cache: bool = True,
) -> int:
    """
    Run mean-edge baseline comparison over several outer folds.

    Скор каждого fold считается со знаком, выбранным по train pool этого fold.
    CI считаются по всем fold: pooled (один bootstrap по субъектам всех fold),
    macro (среднее AUC fold) и по каждому fold отдельно.

    Parameters
    ----------
    data_root : Path
        Корень архива HCP.
    out_dir : Path
        Каталог отчёта.
    tasks : Sequence[str]
        Задачи HCP.
    outer_folds : Sequence[int], optional
        Outer fold (test этих fold объединяются в pooled).
    n_bootstrap : int, optional
        Число bootstrap-реплик.
    seed : int, optional
        Random seed.
    reference : Optional[str], optional
        Эталон основного семейства гипотез.
    alpha : float, optional
        Уровень значимости.
    workers : int, optional
        Потоки чтения матриц.
    use_cache : bool, optional
        Кэшировать скоры в ``out_dir/cache``.

    Returns
    -------
    int
        0 при успехе.
    """
    unknown = sorted(set(tasks) - set(TASKS))
    if unknown:
        raise ValueError(f"Неизвестные задачи {unknown}, допустимы {TASKS}")
    if n_bootstrap < 100:
        raise ValueError(f"n_bootstrap слишком мал: {n_bootstrap}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha должен быть в (0, 1), получено {alpha}")
    folds = normalize_folds(outer_folds)

    root = Path(resolve_existing_path(str(data_root)))
    expected = expected_test_subjects(root / "splits.json", folds)
    cache_dir = out_dir / "cache" if use_cache else None
    cells: List[CellResult] = []
    results: List[CvTaskComparison] = []
    for task in tasks:
        preds_by_fold = {}
        for fold in folds:
            fold_cells = []
            for variant in VARIANTS:
                frame = collect_scores(root, variant, task, fold, workers, cache_dir)
                cell = replace(build_cell(frame, variant, task), outer_fold=fold)
                log.info(
                    "fold %d %s/%s: знак %+d, train AUC %.4f, вырожден=%s",
                    fold, variant, task, cell.sign, cell.train_auc, cell.degenerate,
                )
                fold_cells.append(cell)
            cells.extend(fold_cells)
            preds_by_fold[fold] = {c.variant: c.test for c in fold_cells}
        validate_fold_subjects(preds_by_fold, expected)
        results.append(compare_task_cv(task, preds_by_fold, n_bootstrap=n_bootstrap, seed=seed))

    write_cv_comparison(
        results,
        label=MODEL_NAME,
        folds=folds,
        out_dir=out_dir,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        reference=reference,
        note="Знак скора выбран по train pool каждого fold; без обучения GNN",
        quiet=True,
    )
    diag = diagnostics_table(cells)
    diag.to_csv(out_dir / "cell_diagnostics.csv", index=False)
    report_path = out_dir / "report.md"
    report = report_path.read_text(encoding="utf-8").replace(
        f"# Сравнение вариантов по outer fold: {MODEL_NAME}",
        "# Классификатор «средний вес ребра»: сравнение вариантов по outer fold",
        1,
    )
    report += render_diagnostics(diag)
    report_path.write_text(report, encoding="utf-8")
    print(report)
    log.info("Отчёт записан в %s", out_dir)
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
    parser = argparse.ArgumentParser(
        description="Классификатор «средний вес ребра»: сравнение вариантов с 95% CI",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / ".." / "HPC_data_ref" / "hcp_mmp_mean_ablation_graphs",
    )
    parser.add_argument("--task", action="append", default=None, help="Повторяемый; по умолчанию все 7")
    parser.add_argument(
        "--outer-fold",
        type=int,
        nargs="+",
        default=list(OUTER_FOLDS),
        metavar="K",
        help="Outer fold; test всех указанных fold объединяются в один отчёт (по умолчанию 0 1 2 3)",
    )
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reference",
        default=DEFAULT_REFERENCE,
        help="Эталон основного семейства; 'none' — все пары в одном семействе",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-cache", action="store_true", help="Пересчитать скоры, не читать кэш")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "artifacts" / "mean_edge_baseline",
    )
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
    return run_baseline(
        data_root=args.data_root.resolve(),
        out_dir=args.out_dir.resolve(),
        tasks=tuple(args.task) if args.task else TASKS,
        outer_folds=args.outer_fold,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        reference=None if args.reference.lower() == "none" else args.reference,
        alpha=args.alpha,
        workers=args.workers,
        use_cache=not args.no_cache,
    )


if __name__ == "__main__":
    raise SystemExit(main())
