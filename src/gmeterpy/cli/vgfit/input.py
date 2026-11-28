"""Column mapping and validation for vgfit tie tables."""

from collections.abc import Mapping
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

LEGACY_COLUMNS = {
    "set_id": "runn",
    "h1": "level_1",
    "h2": "level_2",
    "delta_g": "delta_g",
}


def resolve_columns(
    data: pd.DataFrame, overrides: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Use explicit mappings first, followed by Python names and legacy names."""
    overrides = overrides or {}
    return {
        field: overrides.get(field, field if field in data.columns else legacy)
        for field, legacy in LEGACY_COLUMNS.items()
    }


def numeric_column(data: pd.DataFrame, header: str) -> NDArray[np.float64]:
    """Read finite numbers and identify the first invalid cell by source row."""
    column = cast(pd.Series, data[header])
    values = np.asarray(pd.to_numeric(column, errors="coerce"), dtype=float)
    invalid = np.flatnonzero(~np.isfinite(values))
    if invalid.size:
        row = int(invalid[0])
        value = column.iloc[row]
        found = "a missing value" if pd.isna(value) else repr(value)
        raise ValueError(
            f"Column {header!r} requires a finite number in data row {row + 1}. Found {found}."
        )
    return values
