import warnings

import pytest
import pandas as pd
import polars as pl
from data_compare.src.utils import _convert_parameters_to_polars, convert_to_polars


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
