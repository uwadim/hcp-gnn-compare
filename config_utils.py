"""
Утилиты разбора Hydra-конфигурации.

Разрешение путей и опциональных скалярных значений.
"""

# 1. Standard library
import os
from typing import Optional

# 2. External libraries
from hydra.utils import to_absolute_path


def resolve_path(path: object, *, ensure_parent: bool = True) -> Optional[str]:
    """
    Resolve path.

    Разрешает путь относительно оригинальной рабочей директории запуска Hydra.

    Parameters
    ----------
    path : object
        Значение пути из конфигурации.
    ensure_parent : bool, optional
        Создавать родительские директории (для выходных файлов).

    Returns
    -------
    Optional[str]
        Абсолютный путь или ``None``.
    """
    if path is None:
        return None

    path_str = str(path).strip()
    if path_str == "" or path_str.lower() in {"null", "none"}:
        return None

    try:
        resolved = to_absolute_path(path_str)
    except Exception:  # noqa: BLE001
        resolved = os.path.abspath(path_str)

    if ensure_parent:
        parent = os.path.dirname(resolved)
        if parent:
            os.makedirs(parent, exist_ok=True)

    return resolved


def resolve_existing_path(path: object) -> str:
    """
    Resolve existing path.

    Разрешает путь к существующему каталогу/файлу данных без создания родителей.

    Parameters
    ----------
    path : object
        Значение пути из конфигурации.

    Returns
    -------
    str
        Абсолютный путь.

    Raises
    ------
    FileNotFoundError
        Если путь пуст или не существует.
    """
    resolved = resolve_path(path, ensure_parent=False)
    if resolved is None:
        raise FileNotFoundError(f"Путь не задан: {path!r}")
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Путь не найден: {resolved}")
    return resolved


def parse_optional_int(value: object) -> Optional[int]:
    """
    Parse optional int.

    Parameters
    ----------
    value : object
        Значение из конфигурации.

    Returns
    -------
    Optional[int]
        Целое число или ``None``.
    """
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "" or value.lower() in {"null", "none"}:
            return None
        return int(value)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    raise TypeError(f"Не удалось преобразовать значение к int: {value!r}")


def parse_optional_float(value: object) -> Optional[float]:
    """
    Parse optional float.

    Parameters
    ----------
    value : object
        Значение из конфигурации.

    Returns
    -------
    Optional[float]
        Вещественное число или ``None``.
    """
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "" or value.lower() in {"null", "none"}:
            return None
        return float(value)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"Не удалось преобразовать значение к float: {value!r}")


def parse_optional_str(value: object) -> Optional[str]:
    """
    Parse optional str.

    Parameters
    ----------
    value : object
        Значение из конфигурации.

    Returns
    -------
    Optional[str]
        Строка или ``None``.
    """
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "" or value.lower() in {"null", "none"}:
            return None
        return value
    return str(value)
