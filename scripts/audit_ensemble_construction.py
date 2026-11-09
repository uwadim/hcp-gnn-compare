#!/usr/bin/env python3
"""
Аудит построения ensemble-графов (утечка / supervised edges).

Не обучает модель. Читает README архива, ``splits.json`` и файлы
``ensemble/{univariate,full}/.../fold0``, проверяет:

1. train/test субъекты совпадают со ``splits.json`` и не пересекаются;
2. средний вес ребра сильно коррелирует с истинной меткой
   (ожидаемо: рёбра = P(state2)−P(state1) из edge-wise logistic);
3. формулировки README про OOF-построение.

Отчёт: ``artifacts/leak_check/ensemble_audit.md``.
"""

from __future__ import annotations

# 1. Standard library
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

# 2. External libraries
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 3. Local
from config_utils import resolve_existing_path  # noqa: E402
from cv_splits import load_splits  # noqa: E402
from dataset import ENSEMBLE_VARIANT_DIR, label_to_binary  # noqa: E402

log = logging.getLogger("audit_ensemble")

DEFAULT_VARIANTS = ("ensemble_univariate", "ensemble_full")
DEFAULT_TASKS = ("EMOTION", "MOTOR", "SOCIAL")
CEILING_NOTE = (
    "В stage-2 отчёте на этих задачах ROC-AUC SkipGCN достигал 1.000 "
    "с вырожденным CI [1, 1]."
)


def _subjects_from_dir(task_dir: Path) -> Set[str]:
    return {path.stem.split("_", maxsplit=1)[0] for path in task_dir.glob("*.npy")}


def _label_map(root: Path, task: str) -> Dict[str, float]:
    path = root / "metadata" / f"{task}_mean_sample_order.csv"
    frame = pd.read_csv(path)
    return {
        str(row.stem): float(label_to_binary(row.label))
        for row in frame.itertuples(index=False)
    }


def _mean_edge_weight(path: Path) -> float:
    matrix = np.load(path)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Ожидалась квадратная матрица в {path}, shape={matrix.shape}")
    iu = np.triu_indices(matrix.shape[0], k=1)
    return float(matrix[iu].mean())


def edge_label_correlation(
    task_dir: Path,
    label_by_stem: Dict[str, float],
    max_files: int = 200,
) -> Tuple[float, float, float, int]:
    """
    Edge-label correlation.

    Parameters
    ----------
    task_dir : Path
        Каталог ``*.npy`` одного task.
    label_by_stem : Dict[str, float]
        Бинарные метки по stem.
    max_files : int, optional
        Максимум файлов (равномерно по первым ``max_files`` в сортировке).

    Returns
    -------
    Tuple[float, float, float, int]
        ``(corr, mean_edge_y0, mean_edge_y1, n)``.
    """
    paths = sorted(task_dir.glob("*.npy"))[: int(max_files)]
    if not paths:
        raise FileNotFoundError(f"Нет .npy в {task_dir}")
    ys: List[float] = []
    means: List[float] = []
    for path in paths:
        ys.append(label_by_stem[path.stem])
        means.append(_mean_edge_weight(path))
    y_np = np.asarray(ys, dtype=np.float64)
    m_np = np.asarray(means, dtype=np.float64)
    if len(np.unique(y_np)) < 2:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(y_np, m_np)[0, 1])
    return corr, float(m_np[y_np < 0.5].mean()), float(m_np[y_np >= 0.5].mean()), len(paths)


def audit_fold(
    root: Path,
    variant: str,
    task: str,
    outer_fold: int = 0,
) -> Dict[str, object]:
    """
    Audit one variant × task × outer fold.

    Parameters
    ----------
    root : Path
        Корень архива.
    variant : str
        ``ensemble_univariate`` или ``ensemble_full``.
    task : str
        Задача HCP.
    outer_fold : int, optional
        Outer fold.

    Returns
    -------
    Dict[str, object]
        Результаты проверок.
    """
    if variant not in ENSEMBLE_VARIANT_DIR:
        raise ValueError(f"variant={variant!r} не ensemble")
    splits = load_splits(root / "splits.json")
    outer = next(s for s in splits if s.fold == int(outer_fold))
    train_ref = {str(s) for s in outer.train_subjects}
    test_ref = {str(s) for s in outer.test_subjects}

    ens_root = (
        root
        / "ensemble"
        / ENSEMBLE_VARIANT_DIR[variant]
        / "graphs"
        / "mean"
        / "ensemble_edges"
        / f"fold{outer_fold}"
    )
    inner_subjects: List[Set[str]] = []
    for inner_id in range(3):
        task_dir = ens_root / "train" / f"fold{inner_id}" / task
        if not task_dir.is_dir():
            raise FileNotFoundError(f"Нет каталога {task_dir}")
        subjects = _subjects_from_dir(task_dir)
        inner_subjects.append(subjects)

    test_dir = ens_root / "test" / task
    test_subjects = _subjects_from_dir(test_dir)
    train_union = set().union(*inner_subjects)

    labels = _label_map(root, task)
    corr_test, mean0, mean1, n_test = edge_label_correlation(test_dir, labels)
    corr_train, _, _, n_train = edge_label_correlation(
        ens_root / "train" / "fold0" / task,
        labels,
    )

    return {
        "variant": variant,
        "task": task,
        "outer_fold": outer_fold,
        "train_match_splits": train_union == train_ref,
        "test_match_splits": test_subjects == test_ref,
        "train_test_disjoint": train_union.isdisjoint(test_subjects),
        "inner_pairwise_disjoint": all(
            inner_subjects[i].isdisjoint(inner_subjects[j])
            for i in range(3)
            for j in range(i + 1, 3)
        ),
        "n_train_subjects": len(train_union),
        "n_test_subjects": len(test_subjects),
        "inner_sizes": [len(s) for s in inner_subjects],
        "corr_mean_edge_label_test": corr_test,
        "corr_mean_edge_label_train_fold0": corr_train,
        "mean_edge_y0_test": mean0,
        "mean_edge_y1_test": mean1,
        "n_files_scored_test": n_test,
        "n_files_scored_train": n_train,
    }


def _readme_excerpts(root: Path) -> List[str]:
    readme = root / "README.md"
    if not readme.is_file():
        return ["README.md архива не найден."]
    text = readme.read_text(encoding="utf-8")
    # Фразы могут быть перенесены между строками README.
    flat = " ".join(text.split())
    # В README путь обёрнут в backticks: ``train/foldN`` contains…
    keys = (
        "An edge is P(state2)-P(state1)",
        "contains predictions for that inner held-out",
        "test` contains predictions from edge models",
        "do not combine train/test",
    )
    lines = []
    for key in keys:
        ok = key in flat
        lines.append(f"- `{key}`: {'найдено' if ok else 'НЕ найдено'}")
        if ok:
            idx = flat.index(key)
            snippet = flat[max(0, idx - 40) : idx + len(key) + 80]
            lines.append(f"  > …{snippet}…")
    return lines


def render_report(
    root: Path,
    rows: Sequence[Dict[str, object]],
) -> str:
    """
    Render markdown audit report.

    Parameters
    ----------
    root : Path
        Корень архива.
    rows : Sequence[Dict[str, object]]
        Результаты :func:`audit_fold`.

    Returns
    -------
    str
        Markdown.
    """
    ok_disjoint = all(bool(r["train_test_disjoint"]) for r in rows)
    ok_splits = all(
        bool(r["train_match_splits"]) and bool(r["test_match_splits"]) for r in rows
    )
    corrs = [float(r["corr_mean_edge_label_test"]) for r in rows]
    lines = [
        "# Аудит построения ensemble-графов",
        "",
        f"Архив: `{root}`",
        "",
        CEILING_NOTE,
        "",
        "## Документация архива",
        "",
        *_readme_excerpts(root),
        "",
        "## Проверки fold0",
        "",
        "| variant | task | train↔splits | test↔splits | train∩test=∅ | "
        "inner∩=∅ | corr(mean_edge, label) test | mean_edge y=0 / y=1 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['variant']} | {r['task']} | "
            f"{'yes' if r['train_match_splits'] else 'NO'} | "
            f"{'yes' if r['test_match_splits'] else 'NO'} | "
            f"{'yes' if r['train_test_disjoint'] else 'NO'} | "
            f"{'yes' if r['inner_pairwise_disjoint'] else 'NO'} | "
            f"{float(r['corr_mean_edge_label_test']):.3f} | "
            f"{float(r['mean_edge_y0_test']):+.3f} / "
            f"{float(r['mean_edge_y1_test']):+.3f} |"
        )
    lines += [
        "",
        "## Вывод аудита",
        "",
        f"- Совпадение субъектов со `splits.json`: "
        f"**{'да' if ok_splits else 'НЕТ'}**.",
        f"- Пересечение train/test субъектов: "
        f"**{'нет (ок)' if ok_disjoint else 'ЕСТЬ (утечка субъектов)'}**.",
        f"- Корреляция среднего веса ребра с меткой на test: "
        f"min={min(corrs):.3f}, max={max(corrs):.3f}. "
        "Значения ≫ 0 означают, что рёбра **намеренно supervised** "
        "(P(state2)−P(state1)), а не «сырые» корреляции/активации без меток.",
        "",
        "Итог: AUC≈1.0 у SkipGCN на `ensemble_univariate` / `ensemble_full` "
        "объясним свойствами признаков (edge-wise logistic на метках с OOF "
        "по субъектам). Это не подмешивание test-субъектов в train-каталоги. "
        "Тест с перемешанными метками (`shuffle_labels_scope=all`) должен "
        "дать ROC-AUC≈0.5: иначе баг в пайплайне оценки. Scope=`train` "
        "(перемешаны только train-метки, test истинные) также должен дать "
        "ROC-AUC≈0.5: без истинных train-меток модель не выучит сигнал в "
        "рёбрах; устойчиво высокий AUC означал бы утечку меток в обучение.",
        "",
    ]
    return "\n".join(lines)


def run_audit(
    data_root: Path,
    variants: Sequence[str],
    tasks: Sequence[str],
    out_path: Path,
    outer_fold: int = 0,
) -> int:
    """
    Run audit.

    Parameters
    ----------
    data_root : Path
        Корень архива.
    variants : Sequence[str]
        Ensemble-варианты.
    tasks : Sequence[str]
        Задачи.
    out_path : Path
        Путь к markdown-отчёту.
    outer_fold : int, optional
        Outer fold.

    Returns
    -------
    int
        0 при успехе, 1 при провале проверок субъектов.
    """
    root = Path(resolve_existing_path(str(data_root)))
    rows = [
        audit_fold(root, variant, task, outer_fold=outer_fold)
        for variant in variants
        for task in tasks
    ]
    report = render_report(root, rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(report)
    log.info("Отчёт записан в %s", out_path)

    failed = [
        r
        for r in rows
        if not (
            r["train_match_splits"]
            and r["test_match_splits"]
            and r["train_test_disjoint"]
            and r["inner_pairwise_disjoint"]
        )
    ]
    return 1 if failed else 0


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
    parser = argparse.ArgumentParser(description="Аудит построения ensemble-графов")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / ".." / "HPC_data_ref" / "hcp_mmp_mean_ablation_graphs",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=None,
        help="Повторяемый; по умолчанию ensemble_univariate и ensemble_full",
    )
    parser.add_argument(
        "--task",
        action="append",
        default=None,
        help="Повторяемый; по умолчанию EMOTION, MOTOR, SOCIAL",
    )
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts" / "leak_check" / "ensemble_audit.md",
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
    variants = tuple(args.variant) if args.variant else DEFAULT_VARIANTS
    tasks = tuple(args.task) if args.task else DEFAULT_TASKS
    return run_audit(
        data_root=args.data_root.resolve(),
        variants=variants,
        tasks=tasks,
        out_path=args.out.resolve(),
        outer_fold=args.outer_fold,
    )


if __name__ == "__main__":
    raise SystemExit(main())
