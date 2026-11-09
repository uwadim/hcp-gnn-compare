"""
CV splits.

Разборы subject-level сплитов из ``splits.json`` архива HCP.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class OuterFoldSplit:
    """
    Outer fold split.

    Attributes
    ----------
    fold : int
        Номер outer fold (0..3).
    train_subjects : Tuple[str, ...]
        Субъекты обучающей части.
    test_subjects : Tuple[str, ...]
        Субъекты тестовой части.
    inner : Tuple[Dict[str, Any], ...]
        Вложенные inner-сплиты (каждый с ключами fold/train/test).
    """

    fold: int
    train_subjects: Tuple[str, ...]
    test_subjects: Tuple[str, ...]
    inner: Tuple[Dict[str, Any], ...]


def _as_subject_str(value: Any) -> str:
    return str(value).strip()


def load_splits(splits_path: Path | str) -> List[OuterFoldSplit]:
    """
    Load splits.

    Читает ``splits.json`` и возвращает список outer fold.

    Parameters
    ----------
    splits_path : Path | str
        Путь к ``splits.json``.

    Returns
    -------
    List[OuterFoldSplit]
        Четыре outer fold.
    """
    path = Path(splits_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if not isinstance(raw, list) or len(raw) == 0:
        raise ValueError(f"splits.json должен быть непустым списком: {path}")

    folds: List[OuterFoldSplit] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Элемент splits.json должен быть словарём")
        fold_id = int(item["fold"])
        train = tuple(_as_subject_str(s) for s in item["train"])
        test = tuple(_as_subject_str(s) for s in item["test"])
        inner_raw = item.get("inner", [])
        if not isinstance(inner_raw, list):
            raise ValueError(f"inner fold {fold_id} должен быть списком")
        inner: List[Dict[str, Any]] = []
        for inner_item in inner_raw:
            if not isinstance(inner_item, dict):
                raise ValueError("inner элемент должен быть словарём")
            inner.append(
                {
                    "fold": int(inner_item["fold"]),
                    "train": tuple(_as_subject_str(s) for s in inner_item["train"]),
                    "test": tuple(_as_subject_str(s) for s in inner_item["test"]),
                }
            )
        folds.append(
            OuterFoldSplit(
                fold=fold_id,
                train_subjects=train,
                test_subjects=test,
                inner=tuple(inner),
            )
        )
    return folds


def assert_disjoint(train: Sequence[str], test: Sequence[str], name: str) -> None:
    """
    Assert disjoint.

    Проверяет отсутствие пересечения train и test.

    Parameters
    ----------
    train : Sequence[str]
        Обучающие субъекты.
    test : Sequence[str]
        Тестовые субъекты.
    name : str
        Имя сплита для сообщения об ошибке.
    """
    overlap = set(train) & set(test)
    if overlap:
        sample = sorted(overlap)[:5]
        raise ValueError(
            f"{name}: пересечение train/test, примеры: {sample}"
        )


def validate_outer_folds(folds: Sequence[OuterFoldSplit]) -> None:
    """
    Validate outer folds.

    Parameters
    ----------
    folds : Sequence[OuterFoldSplit]
        Outer fold сплиты.
    """
    if len(folds) < 2:
        raise ValueError(f"Ожидалось >= 2 outer fold, получено {len(folds)}")
    for fold in folds:
        assert_disjoint(fold.train_subjects, fold.test_subjects, f"outer_{fold.fold}")
        for inner in fold.inner:
            assert_disjoint(
                list(inner["train"]),
                list(inner["test"]),
                f"outer_{fold.fold}_inner_{inner['fold']}",
            )


def subjects_to_set(subjects: Sequence[str]) -> set[str]:
    """
    Subjects to set.

    Parameters
    ----------
    subjects : Sequence[str]
        Список субъектов.

    Returns
    -------
    set[str]
        Множество строковых id.
    """
    return {_as_subject_str(s) for s in subjects}


def filter_indices_by_subjects(
    subjects: Sequence[str],
    keep: Mapping[str, Any] | set[str],
) -> List[int]:
    """
    Filter indices by subjects.

    Parameters
    ----------
    subjects : Sequence[str]
        Субъект каждого образца.
    keep : Mapping[str, Any] | set[str]
        Множество (или mapping-ключи) разрешённых субъектов.

    Returns
    -------
    List[int]
        Индексы образцов.
    """
    keep_set = set(keep) if not isinstance(keep, set) else keep
    keep_set = {_as_subject_str(s) for s in keep_set}
    return [
        i
        for i, subject in enumerate(subjects)
        if _as_subject_str(subject) in keep_set
    ]


def select_outer_folds(
    folds: Sequence[OuterFoldSplit],
    outer_fold: Optional[int],
) -> List[OuterFoldSplit]:
    """
    Select outer folds.

    Parameters
    ----------
    folds : Sequence[OuterFoldSplit]
        Все outer fold.
    outer_fold : Optional[int]
        Конкретный fold или ``None`` (все).

    Returns
    -------
    List[OuterFoldSplit]
        Выбранные fold.
    """
    if outer_fold is None:
        return list(folds)
    for fold in folds:
        if fold.fold == int(outer_fold):
            return [fold]
    raise ValueError(
        f"outer_fold={outer_fold} не найден среди {[f.fold for f in folds]}"
    )
