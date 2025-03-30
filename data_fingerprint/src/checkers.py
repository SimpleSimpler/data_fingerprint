from typing import Any, Callable
import json

import pandas as pd
import polars as pl


def _raise_same_column_names(argument: Any, **kwargs) -> None:
    """
    Check if the argument is a `pandas.DataFrame` and has more then one occurance of the same column name.
    If so, raise a ValueError.

    > Note: The interesting thing is that pandas allows to have multiple columns with the same name.

    Parameters:
        argument (Any): The argument to check.

    Raises:
        ValueError: If the argument is a `pandas.DataFrame` and has more then one occurance of the same column name.

    Returns:
        None
    """
    if not isinstance(argument, pd.DataFrame):
        return

    df: pd.DataFrame = argument
    column_names_list: list[str] = list(df.columns)
    column_names_set: set[str] = set(column_names_list)

    if len(column_names_list) == len(column_names_set):
        return

    columns_with_same_name: dict[str, int] = {
        column_name: column_names_list.count(column_name)
        for column_name in column_names_set
    }
    raise ValueError(
        "Column names are not unique. Distribution of column names: "
        f"{json.dumps(columns_with_same_name, indent=4)}"
    )


def _raise_hash_column_name(argument: Any, **kwargs) -> None:
    """
    Check if the argument is a `pandas.DataFrame` or `polars.DataFrame` and has a column named `hash`.
    If so, raise a `ValueError`.

    Parameters:
        argument (Any): The argument to check.

    Raises:
        ValueError: If the argument is a `DataFrame` and has a column named `hash`.

    Returns:
        None
    """
    if not isinstance(argument, pd.DataFrame) and not isinstance(
        argument, pl.DataFrame
    ):
        return

    column_names_list: list[str] = list(argument.columns)
    column_names_set: set[str] = set(column_names_list)

    if "hash" in column_names_set:
        raise ValueError("Column names cannot contain 'hash'")


def _raise_source_column_name(argument: Any, **kwargs) -> None:
    """
    Check if the argument is a `pandas.DataFrame` or `polars.DataFrame` and has a column named `source`.
    If so, raise a `ValueError`.

    Parameters:
        argument (Any): The argument to check.

    Raises:
        ValueError: If the argument is a `DataFrame` and has a column named `source`.

    Returns:
        None
    """
    if not isinstance(argument, pd.DataFrame) and not isinstance(
        argument, pl.DataFrame
    ):
        return

    column_names_list: list[str] = list(argument.columns)
    column_names_set: set[str] = set(column_names_list)
    if "source" in column_names_set:
        raise ValueError("Column names cannot contain 'source'")


def _raise_source_names(argument: Any, source_names: list[str]) -> None:
    """
    Check if the argument is a string (*meaning it is a source name*) and if it is already in the list of source names.
    If so, raise a `ValueError`.

    Parameters:
        argument (Any): The argument to check.
        source_names (list[str]): The list of already existing source names.

    Raises:
        ValueError: If the argument is a string and is already in the list of source names.

    Returns:
        None
    """
    if not isinstance(argument, str):
        return

    if argument in source_names:
        raise ValueError(f"Source name already exists: {argument}")

    source_names.append(argument)


_rules_for_inputs: list[Callable[[Any], None]] = [
    _raise_same_column_names,
    _raise_hash_column_name,
    _raise_source_column_name,
    _raise_source_names,
]
"""Rules for checking the inputs, they are applied in the order they are defined."""


def check_inputs(func) -> None:
    """
    Decorator for checking the input arguments against a set of input rules.
    It is developed to be used for functions that take dataframes and source names as inputs.

    Something like this:

    ```python
    @check_inputs
    def func(
        source_0: pl.DataFrame,
        source_1: pl.DataFrame,
        source_0_name: str,
        source_1_name: str,
    ) -> None:
        pass
    ```
    If you look at the :mod:`data_compare.src.comparator` you will see that this decorator is used there on
    all the functions that take dataframes and source names as inputs.

    If any rule is violated, an exception will be raised.
    The rules are defined in the _rules_for_inputs list.

    .. note::
        The rules are applied in the order they are defined. There are rules:
        - there must not be any duplicate column names in the dataframes.
        - there must not be any column named "hash" in the dataframes
        - there must not be any column named "source" in the dataframes
        - there must not be any duplicate source names

    Parameters:
        *args (Any): The inputs to check.

    Raises:
        Exception: If any rule is violated.

    Returns:
        None
    """

    def wrapper(*args, **kwargs):
        source_names: list[str] = []
        for arg in args:
            for rule in _rules_for_inputs:
                rule(arg, source_names=source_names)

        for value in kwargs.values():
            for rule in _rules_for_inputs:
                rule(value, source_names=source_names)
        return func(*args, **kwargs)

    return wrapper
