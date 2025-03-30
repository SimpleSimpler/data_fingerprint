import pytest
import pandas as pd
import polars as pl

from data_compare.src.checkers import check_inputs


def test_same_column_names():
    # cehck only pandas.DataFrame because polars.DataFrame does not allow duplicate column names
    @check_inputs
    def func(
        dataframe_0: pd.DataFrame,
        dataframe_1: pd.DataFrame,
        source_0: str,
        source_1: str,
    ):
        pass

    dataframe_duplicated: pd.DataFrame = pd.DataFrame({"a": [1, 2, 3]})
    tmp_dataframe: pd.DataFrame = pd.DataFrame({"a": [1, 2, 3]})
    dataframe_duplicated = pd.concat([tmp_dataframe, dataframe_duplicated], axis=1)

    dataframe_normal: pd.DataFrame = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    with pytest.raises(ValueError, match=".*Column names are not unique.*"):
        func(
            dataframe_0=dataframe_duplicated,
            dataframe_1=dataframe_normal,
            source_0="source_0",
            source_1="source_1",
        )

    with pytest.raises(ValueError, match=".*Column names are not unique.*"):
        func(
            dataframe_0=dataframe_normal,
            dataframe_1=dataframe_duplicated,
            source_0="source_0",
            source_1="source_1",
        )

    with pytest.raises(ValueError, match=".*Column names are not unique.*"):
        func(
            dataframe_duplicated,
            dataframe_normal,
            "source_0",
            "source_1",
        )

    with pytest.raises(ValueError, match=".*Column names are not unique.*"):
        func(
            dataframe_normal,
            dataframe_duplicated,
            "source_0",
            "source_1",
        )


def test_hash_column_name() -> None:
    @check_inputs
    def func(
        dataframe_0: pd.DataFrame,
        dataframe_1: pd.DataFrame,
        source_0: str,
        source_1: str,
    ) -> None:
        pass

    tmp_dataframe: pd.DataFrame = pd.DataFrame({"hash": [1, 2, 3], "b": [4, 5, 6]})
    normal_dataframe: pd.DataFrame = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    with pytest.raises(ValueError, match=".*Column names cannot contain 'hash'.*"):
        func(
            dataframe_0=tmp_dataframe,
            dataframe_1=normal_dataframe,
            source_0="source_0",
            source_1="source_1",
        )
    with pytest.raises(ValueError, match=".*Column names cannot contain 'hash'.*"):
        func(
            dataframe_0=normal_dataframe,
            dataframe_1=tmp_dataframe,
            source_0="source_0",
            source_1="source_1",
        )

    with pytest.raises(ValueError, match=".*Column names cannot contain 'hash'.*"):
        func(
            pl.from_dataframe(tmp_dataframe),
            pl.from_dataframe(normal_dataframe),
            "source_0",
            "source_1",
        )
    with pytest.raises(ValueError, match=".*Column names cannot contain 'hash'.*"):
        func(
            pl.from_dataframe(normal_dataframe),
            pl.from_dataframe(tmp_dataframe),
            "source_0",
            "source_1",
        )


def test_source_column_name() -> None:
    @check_inputs
    def func(
        dataframe_0: pd.DataFrame,
        dataframe_1: pd.DataFrame,
        source_0: str,
        source_1: str,
    ) -> None:
        pass

    tmp_dataframe: pd.DataFrame = pd.DataFrame({"source": [1, 2, 3], "b": [4, 5, 6]})
    normal_dataframe: pd.DataFrame = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    with pytest.raises(ValueError, match=".*Column names cannot contain 'source'.*"):
        func(
            dataframe_0=tmp_dataframe,
            dataframe_1=normal_dataframe,
            source_0="source_0",
            source_1="source_1",
        )
    with pytest.raises(ValueError, match=".*Column names cannot contain 'source'.*"):
        func(
            dataframe_0=normal_dataframe,
            dataframe_1=tmp_dataframe,
            source_0="source_0",
            source_1="source_1",
        )

    with pytest.raises(ValueError, match=".*Column names cannot contain 'source'.*"):
        func(
            pl.from_dataframe(tmp_dataframe),
            pl.from_dataframe(normal_dataframe),
            "source_0",
            "source_1",
        )
    with pytest.raises(ValueError, match=".*Column names cannot contain 'source'.*"):
        func(
            pl.from_dataframe(normal_dataframe),
            pl.from_dataframe(tmp_dataframe),
            "source_0",
            "source_1",
        )


def test_source_names() -> None:
    @check_inputs
    def func(
        source_0: str,
        source_1: str,
    ) -> None:
        pass

    with pytest.raises(ValueError, match=".*Source name already exists.*"):
        func(
            source_0="source_0",
            source_1="source_0",
        )
