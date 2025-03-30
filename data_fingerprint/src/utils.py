import warnings
from typing import Callable, Any

import polars as pl
import pandas as pd

from data_fingerprint.src.models import RowDifference, DataReport


def _convert_parameters_to_polars(*args, **kwargs) -> tuple[tuple, dict]:
    """
    Convert pandas DataFrames to polars DataFrames.

    Raises:
        UserWarning: If a pandas DataFrame is found.

    Args:
        *args: The arguments to convert.
        **kwargs: The keyword arguments to convert.

    Returns:
        tuple[tuple, dict]: The converted arguments and keyword arguments.
    """

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


def convert_to_polars(func: Callable) -> Callable:
    """
    Decorator to convert pandas DataFrames to polars DataFrames.
    If the DataFrame is already a polars DataFrame, it will be returned as is.

    This decorator is useful when you want to use a function that expects polars DataFrames,
    but you have pandas DataFrames.
    It will convert the pandas DataFrames to polars DataFrames before calling the function.
    It will also warn the user that the DataFrames are being converted.
    It is recommended to convert the DataFrames to polars before analyzing and finding the differences.
    This will avoid any data type changes that may occur during the conversion.
    The conversion is done using the `pl.from_pandas` function.

    Args:
        func (Callable): The function to decorate.

    Returns:
        Callable: The decorated function.
    """

    def wrapper(*args, **kwargs) -> pl.DataFrame:
        arg, kwa = _convert_parameters_to_polars(*args, **kwargs)
        return func(*arg, **kwa)

    return wrapper


def convert_row_differences_to_pandas(
    row_differences: list[RowDifference],
) -> pl.DataFrame:
    """
    Convert a list of :class:`data_compare.src.models.RowDifference` objects to a pandas DataFrame.

    Args:
        row_differences (list[:class:`data_compare.src.models.RowDifference`]): The list of :class:`data_compare.src.models.RowDifference` objects.

    Returns:
        pl.DataFrame: The resulting DataFrame.
        If the list is empty, returns an empty DataFrame.
        The DataFrame will have the same columns as the :class:`data_compare.src.models.RowDifference` objects.
        The DataFrame will have an additional column "source" with the source of the row.
    """
    row_differences_list: list[pl.DataFrame] = []

    for row_diff in row_differences:
        df: pl.DataFrame = pl.DataFrame(row_diff.row)
        df = df.with_columns(pl.lit(row_diff.source).alias("source"))
        row_differences_list.append(df)

    if len(row_differences_list) == 0:
        return pl.DataFrame()

    return pl.concat(row_differences_list)


def get_dataframe(data_report: DataReport) -> pl.DataFrame:
    """
    Convert a :class:`data_compare.src.models.DataReport` object to a pandas DataFrame.

    This function will convert the row differences to a pandas DataFrame.
    The DataFrame will have only columns that are comparable between the two dataframes (*more information in :class:`data_compare.src.models.DataReport`*).

    .. note::
       This function will return an empty DataFrame if there are no row differences.

    Args:
        data_report (:class:`data_compare.src.models.DataReport`): The :class:`data_compare.src.models.DataReport` object.

    Returns:
        pl.DataFrame: The resulting DataFrame.
    """
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
