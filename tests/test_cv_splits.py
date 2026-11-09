"""Тесты разбора splits.json."""

from __future__ import annotations

from pathlib import Path

import pytest

from cv_splits import (
    assert_disjoint,
    load_splits,
    select_outer_folds,
    validate_outer_folds,
)


def test_assert_disjoint_ok() -> None:
    """Пересечения нет — без исключения."""
    assert_disjoint(["a", "b"], ["c"], "demo")


def test_assert_disjoint_raises() -> None:
    """Пересечение train/test вызывает ошибку."""
    with pytest.raises(ValueError, match="пересечение"):
        assert_disjoint(["a", "b"], ["b", "c"], "demo")


def test_load_splits(data_root: Path) -> None:
    """splits.json: 4 fold, без пересечений."""
    folds = load_splits(data_root / "splits.json")
    assert len(folds) == 4
    validate_outer_folds(folds)
    assert len(folds[0].train_subjects) == 624
    assert len(folds[0].test_subjects) == 208
    assert len(folds[0].inner) == 3


def test_select_outer_folds(data_root: Path) -> None:
    """Выбор одного outer fold."""
    folds = load_splits(data_root / "splits.json")
    selected = select_outer_folds(folds, 2)
    assert len(selected) == 1
    assert selected[0].fold == 2
    with pytest.raises(ValueError):
        select_outer_folds(folds, 99)
