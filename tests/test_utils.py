import warnings

import pytest
import pandas as pd
import polars as pl

from data_fingerprint.src.comparator import get_data_report
from data_fingerprint.src.utils import (
    _convert_parameters_to_polars,
    convert_to_polars,
    get_column_difference_ratio,
    get_dataframe,
    get_number_of_differences_per_source,
    get_number_of_row_differences,
    get_ratio_of_differences_per_source,
)


def test_convert_parameters_to_polars() -> None:
    args: tuple = (
        pd.DataFrame({"a": [1, 2, 3]}),
        pd.DataFrame({"b": [4, 5, 6]}),
        1,
        2,
        3,
    )
    kwargs: dict = {
        "df1": pd.DataFrame({"c": [7, 8, 9]}),
        "df2": pd.DataFrame({"d": [10, 11, 12]}),
    }

    with pytest.warns(
        UserWarning, match=".*Trasnforming pandas DataFrames to polars DataFrames.*"
    ):
        converted_args, converted_kwargs = _convert_parameters_to_polars(
            *args, **kwargs
        )

    assert isinstance(converted_args[0], pl.DataFrame)
    assert isinstance(converted_args[1], pl.DataFrame)
    assert converted_args[2:] == [1, 2, 3]
    assert isinstance(converted_kwargs["df1"], pl.DataFrame)
    assert isinstance(converted_kwargs["df2"], pl.DataFrame)


def test_convert_to_polars() -> None:
    df1 = pl.DataFrame({"a": [1, 2, 3]})
    df2 = pl.DataFrame({"b": [4, 5, 6]})
    c = 7

    @convert_to_polars
    def dummy_function(*args, **kwargs):
        return args, kwargs

    converted_args, converted_kwargs = dummy_function(df1, df2, c=c)

    assert converted_args[0].equals(df1)
    assert converted_args[1].equals(df2)
    assert converted_kwargs == {"c": c}


def test_get_column_difference_ratio():
    df0 = pl.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [5, 6, 7, 8],
            "c": [9, 10, 11, 12],
            "d": [13, 14, 15, 16],
        }
    )
    df1 = pl.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [5, 6, 77, 8],
            "c": [99, 10, 111, 122],
            "d": [13, 14, 15, 16],
        }
    )
    df0_name = "df0"
    df1_name = "df1"
    report = get_data_report(df0, df1, df0_name, df1_name, ["a"])
    assert get_column_difference_ratio(report) == {
        "a": 0.0,
        "b": 0.25,
        "c": 0.75,
        "d": 0.0,
    }


def test_get_column_no_difference_report():
    df0 = pl.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [5, 6, 7, 8],
            "c": [9, 10, 11, 12],
            "d": [13, 14, 15, 16],
        }
    )
    df1 = df0
    df0_name = "df0"
    df1_name = "df1"
    report = get_data_report(df0, df1, df0_name, df1_name, ["a"])

    with pytest.warns(UserWarning):
        assert get_column_difference_ratio(report) == {
            "a": 0.0,
            "b": 0.0,
            "c": 0.0,
            "d": 0.0,
        }
    assert get_number_of_differences_per_source(report) == {
        "df0": 0,
        "df1": 0,
    }
    assert get_number_of_row_differences(report) == 0

    with pytest.warns(UserWarning):
        assert get_ratio_of_differences_per_source(report) == {
            "df0": 0.0,
            "df1": 0.0,
        }


def test_no_same_columns():
    df0 = pl.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [5, 6, 7, 8],
            "c": [9, 10, 11, 12],
            "d": [13, 14, 15, 16],
        }
    )
    df1 = pl.DataFrame(
        {
            "e": [1, 2, 3, 4],
            "f": [5, 6, 7, 8],
            "g": [9, 10, 11, 12],
            "h": [13, 14, 15, 16],
        }
    )
    df0_name = "df0"
    df1_name = "df1"
    report = get_data_report(df0, df1, df0_name, df1_name)
    with pytest.warns(UserWarning):
        assert get_dataframe(report).shape == (0, 0)

    assert get_number_of_row_differences(report) == 8
    assert get_number_of_differences_per_source(report) == {
        "df0": 4,
        "df1": 4,
    }
    assert get_ratio_of_differences_per_source(report) == {
        "df0": 0.5,
        "df1": 0.5,
    }
    with pytest.warns(UserWarning):
        assert get_column_difference_ratio(report) == {}


def test_column_ratio_differences():
    df0 = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 100],
            "b": [5, 50, 7, 8, 100],
            "c": [9, 10, 11, 12, 100],
            "d": [13, 14, 15, 16, 100],
        }
    )
    df1 = pl.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [5, 6, 7, 8],
            "c": [9, 50, 50, 12],
            "d": [13, 14, 15, 16],
        }
    )
    df0_name = "df0"
    df1_name = "df1"
    # 14
    # a = 2/14, b= 4/14, c= 6/14, d= 2/14
    report = get_data_report(df0, df1, df0_name, df1_name, grouping_columns=["a"])
    column_difference_ratio = get_column_difference_ratio(report)
    assert column_difference_ratio == {
        "a": 2 / 14,
        "b": 4 / 14,
        "c": 6 / 14,
        "d": 2 / 14,
    }
    assert abs(sum([x for x in column_difference_ratio.values()]) - 1) < 1e-5


def test_column_ratio_differences_duplicates():
    df0 = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 100, 5, 5, 6],
            "b": [5, 50, 7, 8, 100, 5, 5, 6],
            "c": [9, 10, 11, 12, 100, 5, 5, 6],
            "d": [13, 14, 15, 16, 100, 5, 5, 6],
        }
    )
    df1 = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 6, 6],
            "b": [5, 6, 7, 8, 6, 6],
            "c": [9, 50, 50, 12, 6, 6],
            "d": [13, 14, 15, 16, 6, 6],
        }
    )
    df0_name = "df0"
    df1_name = "df1"
    # 14
    # a = 8/38, b= 10/38, c= 12/38, d= 8/38
    report = get_data_report(df0, df1, df0_name, df1_name, grouping_columns=["a"])
    column_difference_ratio = get_column_difference_ratio(report)
    assert column_difference_ratio == {
        "a": 8 / 38,
        "b": 10 / 38,
        "c": 12 / 38,
        "d": 8 / 38,
    }
    assert abs(sum([x for x in column_difference_ratio.values()]) - 1) < 1e-5


def test_column_ratio_differences_same():
    df0 = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 100, 5, 5, 6],
            "b": [5, 50, 7, 8, 100, 5, 5, 6],
            "c": [9, 10, 11, 12, 100, 5, 5, 6],
            "d": [13, 14, 15, 16, 100, 5, 5, 6],
        }
    )
    df1 = df0
    df0_name = "df0"
    df1_name = "df1"
    # 14
    # a = 8/38, b= 10/38, c= 12/38, d= 8/38
    report = get_data_report(df0, df1, df0_name, df1_name, grouping_columns=["a"])
    with pytest.warns(UserWarning):
        column_difference_ratio = get_column_difference_ratio(report)
    assert column_difference_ratio == {
        "a": 0.0,
        "b": 0.0,
        "c": 0.0,
        "d": 0.0,
    }
    assert sum([x for x in column_difference_ratio.values()]) == 0.0


def test_column_ratio_differences_without_grouping():
    df0 = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 100],
            "b": [5, 50, 7, 8, 100],
            "c": [9, 10, 11, 12, 100],
            "d": [13, 14, 15, 16, 100],
        }
    )
    df1 = pl.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [5, 6, 7, 8],
            "c": ["a", "b", "c", "d"],
            "d": [13, 14, 15, 16],
        }
    )
    df0_name = "df0"
    df1_name = "df1"
    report = get_data_report(df0, df1, df0_name, df1_name)
    column_difference_ratio = get_column_difference_ratio(report)
    assert column_difference_ratio == {
        "a": 6 / 18,
        "b": 6 / 18,
        "d": 6 / 18,
    }
    assert abs(sum([x for x in column_difference_ratio.values()]) - 1) < 1e-5


def test_column_ratio_differences_duplicates_without_grouping():
    df0 = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 100, 5, 5, 6],
            "b": [5, 50, 7, 8, 100, 5, 5, 6],
            "c": [9, 10, 11, 12, 100, 5, 5, 6],
            "d": [13, 14, 15, 16, 100, 5, 5, 6],
        }
    )
    df1 = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 6, 6],
            "b": [5, 6, 7, 8, 6, 6],
            "c": [9, 50, 50, 12, 6, 6],
            "d": [13, 14, 15, 16, 6, 6],
        }
    )
    df0_name = "df0"
    df1_name = "df1"
    report = get_data_report(df0, df1, df0_name, df1_name)
    column_difference_ratio = get_column_difference_ratio(report)
    assert column_difference_ratio == {
        "a": 4 / 16,
        "b": 4 / 16,
        "c": 4 / 16,
        "d": 4 / 16,
    }
    assert abs(sum([x for x in column_difference_ratio.values()]) - 1) < 1e-5
