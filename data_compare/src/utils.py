import warnings
from typing import Callable, Any

import polars as pl
import pandas as pd

from data_compare.src.models import RowDifference, DataReport


def _convert_parameters_to_polars(*args, **kwargs) -> tuple[tuple, dict]:
    def transform_pandas_to_polars(arg):
        warnings.warn(
            "Trasnforming pandas DataFrames to polars DataFrames. "
            "There may be some data types changes. "
            "Please transform DataFrames to polars before analyzing and "
            "find out the differences.",
            UserWarning,
        )
        return pl.from_pandas(arg)

    arg: list[pl.DataFrame] = [
        transform_pandas_to_polars(arg) if isinstance(arg, pd.DataFrame) else arg
        for arg in args
    ]
    kwa: dict[str, pl.DataFrame] = {
        key: (
            transform_pandas_to_polars(value)
            if isinstance(value, pd.DataFrame)
            else value
        )
        for key, value in kwargs.items()
    }
    return arg, kwa


def convert_to_polars(func: Callable) -> tuple[tuple, dict]:
    def wrapper(*args, **kwargs) -> pl.DataFrame:
        arg, kwa = _convert_parameters_to_polars(*args, **kwargs)
        return func(*arg, **kwa)

    return wrapper


def convert_row_differences_to_pandas(
    row_differences: list[RowDifference],
) -> pl.DataFrame:
    row_differences_list: list[pl.DataFrame] = []

    for row_diff in row_differences:
        df: pl.DataFrame = pl.DataFrame(row_diff.row)
        df = df.with_columns(pl.lit(row_diff.source).alias("source"))
        row_differences_list.append(df)

    if len(row_differences_list) == 0:
        return pl.DataFrame()

    return pl.concat(row_differences_list)


def get_dataframe(data_report: DataReport) -> pl.DataFrame:
    gathered_rows: list[pl.DataFrame] = []

    for rd in data_report.row_differences:
        tmp_rows: pl.DataFrame = pl.DataFrame(rd.row)
        if isinstance(rd, RowDifference):
            tmp_rows = tmp_rows.with_columns(pl.lit(rd.source).alias("source"))
            gathered_rows.append(tmp_rows)
            continue

        tmp_rows: pl.DataFrame = pl.DataFrame(rd.row_with_source)
        gathered_rows.append(tmp_rows)

    if len(gathered_rows) == 0:
        return pl.DataFrame()

    return pl.concat(gathered_rows, how="vertical_relaxed")
