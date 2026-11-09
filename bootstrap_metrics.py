"""
Bootstrap Metrics.

Точечная оценка и 95% доверительный интервал ROC-AUC методом bootstrap.
Поддерживается кластерный bootstrap по субъектам (несколько образцов
на субъекта) и парные реплики для сравнения нескольких скоров на одних
и тех же образцах.
"""

from __future__ import annotations

# 1. Standard library
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

# 2. External libraries
import numpy as np


def subject_from_stem(stem: str) -> str:
    """
    Subject from stem.

    Parameters
    ----------
    stem : str
        Идентификатор образца вида ``{subject}_{run}_{condition}``.

    Returns
    -------
    str
        Идентификатор субъекта.
    """
    subject = str(stem).split("_", maxsplit=1)[0]
    if not subject:
        raise ValueError(f"Не удалось выделить субъекта из stem={stem!r}")
    return subject


def _resample_indices(
    n_samples: int,
    groups: Optional[Sequence[str]],
    n_bootstrap: int,
    seed: int,
) -> List[np.ndarray]:
    """
    Resample indices.

    Parameters
    ----------
    n_samples : int
        Число образцов.
    groups : Optional[Sequence[str]]
        Кластер (субъект) каждого образца; ``None`` — bootstrap по образцам.
    n_bootstrap : int
        Число реплик.
    seed : int
        Random seed.

    Returns
    -------
    List[np.ndarray]
        Индексы образцов для каждой реплики.
    """
    rng = np.random.default_rng(int(seed))
    if groups is None:
        return [rng.integers(0, n_samples, size=n_samples) for _ in range(int(n_bootstrap))]

    groups_np = np.asarray([str(g) for g in groups])
    if groups_np.size != n_samples:
        raise ValueError(f"groups: {groups_np.size} != n_samples: {n_samples}")
    # Сортировка даёт одинаковые реплики для одинаковых наборов субъектов.
    unique = np.unique(groups_np)
    members = [np.flatnonzero(groups_np == g) for g in unique]
    out: List[np.ndarray] = []
    for _ in range(int(n_bootstrap)):
        picked = rng.integers(0, unique.size, size=unique.size)
        out.append(np.concatenate([members[i] for i in picked]))
    return out


def bootstrap_auc_replicates(
    y_true: Sequence[float],
    scores: Mapping[str, Sequence[float]],
    groups: Optional[Sequence[str]] = None,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """
    Paired bootstrap AUC replicates.

    Все скоры оцениваются на одних и тех же репликах, поэтому разности
    реплик дают парный bootstrap.

    Parameters
    ----------
    y_true : Sequence[float]
        Бинарные метки.
    scores : Mapping[str, Sequence[float]]
        Скоры положительного класса по именам (варианты/модели).
    groups : Optional[Sequence[str]], optional
        Субъект каждого образца для кластерного bootstrap.
    n_bootstrap : int, optional
        Число реплик.
    seed : int, optional
        Random seed.

    Returns
    -------
    Tuple[Dict[str, float], Dict[str, np.ndarray]]
        Точечные AUC и массивы реплик (``nan`` — реплика с одним классом).
    """
    y_np = np.asarray(y_true, dtype=np.float64).astype(int)
    if not scores:
        raise ValueError("scores пуст")
    score_np = {name: np.asarray(s, dtype=np.float64) for name, s in scores.items()}
    for name, s in score_np.items():
        if s.shape != y_np.shape:
            raise ValueError(f"scores[{name!r}]: {s.shape} != y_true: {y_np.shape}")
        if not np.all(np.isfinite(s)):
            raise ValueError(f"scores[{name!r}] содержит нечисловые значения")
    if len(np.unique(y_np)) < 2:
        raise ValueError("y_true содержит один класс — AUC не определён")

    point = {name: _roc_auc(y_np, s) for name, s in score_np.items()}
    reps = {name: np.full(int(n_bootstrap), np.nan) for name in score_np}
    for b, idx in enumerate(_resample_indices(y_np.size, groups, n_bootstrap, seed)):
        yt = y_np[idx]
        if len(np.unique(yt)) < 2:
            continue
        for name, s in score_np.items():
            reps[name][b] = _rank_auc(yt, s[idx])
    return point, reps


def percentile_ci(
    replicates: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    Percentile CI.

    Parameters
    ----------
    replicates : np.ndarray
        Bootstrap-реплики статистики.
    alpha : float, optional
        Уровень двустороннего CI (0.05 → 95%).

    Returns
    -------
    Tuple[float, float]
        ``(low, high)``; ``nan`` если реплик нет.
    """
    finite = replicates[np.isfinite(replicates)]
    if finite.size == 0:
        return float("nan"), float("nan")
    low, high = np.percentile(finite, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(low), float(high)


def _studentized_stats(
    estimates: Sequence[float],
    replicates: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Studentized observed and null bootstrap statistics.

    ``t_j = |θ̂_j| / se_j``; нулевое распределение — центрированные реплики
    ``|θ*_bj − θ̂_j| / se_j``. Строки реплик с ``nan`` отбрасываются.

    Parameters
    ----------
    estimates : Sequence[float]
        Точечные оценки ``θ̂_j``, форма ``(m,)``.
    replicates : np.ndarray
        Совместные реплики ``θ*``, форма ``(B, m)``.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``t_obs`` формы ``(m,)`` и ``t_null`` формы ``(B', m)``.

    Raises
    ------
    ValueError
        При несовпадении форм или отсутствии валидных реплик.
    """
    est = np.asarray(estimates, dtype=np.float64)
    reps = np.asarray(replicates, dtype=np.float64)
    if reps.ndim != 2 or reps.shape[1] != est.size:
        raise ValueError(f"replicates: ожидалась форма (B, {est.size}), получено {reps.shape}")
    if not np.all(np.isfinite(est)):
        raise ValueError("estimates содержит нечисловые значения")
    reps = reps[np.all(np.isfinite(reps), axis=1)]
    if reps.shape[0] < 2:
        raise ValueError("Недостаточно валидных bootstrap-реплик")

    se = reps.std(axis=0, ddof=1)
    degenerate = se <= 0.0
    safe_se = np.where(degenerate, 1.0, se)
    # Нулевая дисперсия: оценка без разброса, t = 0 при θ̂ = 0, иначе ∞.
    t_obs = np.where(degenerate, np.where(est == 0.0, 0.0, np.inf), np.abs(est) / safe_se)
    t_null = np.where(degenerate, 0.0, np.abs(reps - est) / safe_se)
    return t_obs, t_null


def studentized_p_values(
    estimates: Sequence[float],
    replicates: np.ndarray,
) -> List[float]:
    """
    Marginal two-sided studentized bootstrap p-values (H0: θ_j = 0).

    Те же статистики, что в :func:`romano_wolf_adjust`, без учёта
    множественности: ``p_j = (1 + #{t*_bj ≥ t_j}) / (B + 1)``.

    Parameters
    ----------
    estimates : Sequence[float]
        Точечные оценки, форма ``(m,)``.
    replicates : np.ndarray
        Совместные реплики, форма ``(B, m)``.

    Returns
    -------
    List[float]
        Сырые p-value.
    """
    t_obs, t_null = _studentized_stats(estimates, replicates)
    counts = np.sum(t_null >= t_obs, axis=0)
    return ((1.0 + counts) / (t_null.shape[0] + 1.0)).tolist()


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    """
    Holm-Bonferroni adjustment.

    Parameters
    ----------
    p_values : Sequence[float]
        Сырые p-value (``nan`` пропускаются).

    Returns
    -------
    List[float]
        Скорректированные p-value в исходном порядке.
    """
    p_np = np.asarray(p_values, dtype=np.float64)
    adjusted = np.full(p_np.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(p_np))
    m = valid.size
    running = 0.0
    for rank, i in enumerate(valid[np.argsort(p_np[valid])]):
        running = max(running, min(1.0, (m - rank) * p_np[i]))
        adjusted[i] = running
    return adjusted.tolist()


def bh_adjust(p_values: Sequence[float]) -> List[float]:
    """
    Benjamini-Hochberg FDR adjustment.

    Parameters
    ----------
    p_values : Sequence[float]
        Сырые p-value (``nan`` пропускаются).

    Returns
    -------
    List[float]
        q-value в исходном порядке.
    """
    p_np = np.asarray(p_values, dtype=np.float64)
    adjusted = np.full(p_np.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(p_np))
    m = valid.size
    if m == 0:
        return adjusted.tolist()
    order = valid[np.argsort(p_np[valid])]
    scaled = p_np[order] * m / np.arange(1, m + 1)
    adjusted[order] = np.minimum(1.0, np.minimum.accumulate(scaled[::-1])[::-1])
    return adjusted.tolist()


def romano_wolf_adjust(
    estimates: Sequence[float],
    replicates: np.ndarray,
) -> List[float]:
    """
    Romano-Wolf studentized step-down adjustment (bootstrap maxT).

    Контролирует FWER при произвольной зависимости гипотез, оценивая её
    по совместным bootstrap-репликам (Romano & Wolf, 2005, 2016).
    ``H0_j: θ_j = 0``, двусторонняя; статистики — как в
    :func:`studentized_p_values`, поэтому всегда ``p_raw ≤ p_RW``, а
    ``p_RW ≤ p_Holm(p_raw)`` — приближённо (студентизация выравнивает
    нулевые распределения гипотез).

    Parameters
    ----------
    estimates : Sequence[float]
        Точечные оценки ``θ̂_j``, форма ``(m,)``.
    replicates : np.ndarray
        Совместные реплики ``θ*``, форма ``(B, m)``; строки с ``nan``
        отбрасываются.

    Returns
    -------
    List[float]
        Скорректированные p-value в исходном порядке.
    """
    t_obs, t_null = _studentized_stats(estimates, replicates)
    order = np.argsort(-t_obs, kind="stable")
    # max по гипотезам order[k:], для каждой реплики.
    tail_max = np.maximum.accumulate(t_null[:, order][:, ::-1], axis=1)[:, ::-1]
    raw = (1.0 + np.sum(tail_max >= t_obs[order], axis=0)) / (t_null.shape[0] + 1.0)
    adjusted = np.empty(t_obs.size)
    adjusted[order] = np.minimum(1.0, np.maximum.accumulate(raw))
    return adjusted.tolist()


def bootstrap_roc_auc_ci(
    y_true: Sequence[float],
    y_score: Sequence[float],
    n_bootstrap: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
    groups: Optional[Sequence[str]] = None,
) -> Tuple[float, float, float]:
    """
    Bootstrap ROC-AUC CI.

    Parameters
    ----------
    y_true : Sequence[float]
        Истинные бинарные метки.
    y_score : Sequence[float]
        Скоры / вероятности положительного класса.
    n_bootstrap : int, optional
        Число bootstrap-реплик.
    seed : int, optional
        Random seed.
    alpha : float, optional
        Уровень для двустороннего CI (0.05 → 95%).
    groups : Optional[Sequence[str]], optional
        Субъект каждого образца; задаёт кластерный bootstrap.

    Returns
    -------
    Tuple[float, float, float]
        ``(auc, ci_low, ci_high)``.
    """
    y_true_np = np.asarray(y_true, dtype=np.float64)
    y_score_np = np.asarray(y_score, dtype=np.float64)
    valid = (
        (y_true_np >= 0.0)
        & (y_true_np <= 1.0)
        & np.isfinite(y_score_np)
    )
    y_true_np = y_true_np[valid].astype(int)
    y_score_np = y_score_np[valid]
    groups_valid = None if groups is None else np.asarray(groups)[valid].tolist()

    if y_true_np.size == 0 or len(np.unique(y_true_np)) < 2:
        return float("nan"), float("nan"), float("nan")

    point, reps = bootstrap_auc_replicates(
        y_true_np,
        {"score": y_score_np},
        groups=groups_valid,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    low, high = percentile_ci(reps["score"], alpha=alpha)
    return point["score"], low, high


def _rank_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    ROC-AUC через статистику Манна–Уитни (средние ранги при связях).

    Parameters
    ----------
    y_true : np.ndarray
        Бинарные метки (оба класса присутствуют).
    y_score : np.ndarray
        Скоры.

    Returns
    -------
    float
        ROC-AUC, совпадает с ``sklearn.metrics.roc_auc_score``.
    """
    from scipy.stats import rankdata

    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = int(y_true.size - n_pos)
    ranks = rankdata(y_score)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")
