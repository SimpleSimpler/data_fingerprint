from typing import Union, Optional

import polars as pl

from data_fingerprint.src.models import (
    ColumnDifference,
    RowDifference,
    RowGroupDifference,
    DataReport,
)
from data_fingerprint.src.utils import (
    convert_to_polars,
    convert_row_differences_to_pandas,
)
from data_fingerprint.src.checkers import check_inputs
from data_fingerprint.src.difference_types import (
    ColumnNameDifferenceType,
    ColumnDataTypeDifferenceType,
    RowDifferenceType,
)


@convert_to_polars
@check_inputs
def get_column_name_differences(
    df0: pl.DataFrame, df1: pl.DataFrame, df0_name: str, df1_name: str
) -> tuple[list[str], list[ColumnDifference]]:
    """
    Get the differences in column names between two `polars.DataFrame`.
    The referent DataFrame is `df0`. The differences are returned as a list of :class:`data_compare.src.models.ColumnDifference` objects.

    Example:
        ```python
        import polars as pl
        from data_compare.src.comparator import get_column_name_differences
        from data_compare.src.models import ColumnDifference
        df0 = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        df1 = pl.DataFrame({"a": [1, 2], "c": [3, 4]})
        df0_name = "df0"
        df1_name = "df1"
        same_columns, different_columns = get_column_name_differences(df0, df1, df0_name, df1_name)
        print(same_columns)
        print(different_columns)
        ```
    Result:
        ```
        {'a'}

        [ColumnDifference(source='df0', column_name='c',
        difference_type=<ColumnNameDifferenceType.MISSING: 'MISSING'>, more_information=None),
        ColumnDifference(source='df0', column_name='b',
        difference_type=<ColumnNameDifferenceType.EXTRA: 'EXTRA'>, more_information=None)]
        ```

    Args:
        df0 (pl.DataFrame): The first DataFrame.
        df1 (pl.DataFrame): The second DataFrame.
        df0_name (str): The name of the first DataFrame.
        df1_name (str): The name of the second DataFrame.

    Returns:
        list[str]: A list of column names that are the same in both DataFrames.

        list[ColumnDifference]: A list of :class:`data_compare.src.models.ColumnDifference` objects representing the differences in column names.
    """
    column_names_0 = set(df0.columns)
    column_names_1 = set(df1.columns)

    # Get the differences in column names
    # Extra columns are columns in df0 that are not in df1
    extra_columns = column_names_0 - column_names_1
    # Missing columns are columns in df1 that are not in df0
    missing_columns = column_names_1 - column_names_0

    # get the same column names
    same_columns = column_names_0 & column_names_1

    column_differences: list[ColumnDifference] = []
    for missing_col in missing_columns:
        column_differences.append(
            ColumnDifference(
                source=df0_name,
                column_name=missing_col,
                difference_type=ColumnNameDifferenceType.MISSING,
            )
        )
    for extra_col in extra_columns:
        column_differences.append(
            ColumnDifference(
                source=df0_name,
                column_name=extra_col,
                difference_type=ColumnNameDifferenceType.EXTRA,
            )
        )
    return same_columns, column_differences


@convert_to_polars
@check_inputs
def get_column_dtype_differences(
    df0: pl.DataFrame, df1: pl.DataFrame, df0_name: str, df1_name: str
) -> tuple[list[str], list[ColumnDifference]]:
    """
    Get the differences in column types between two `polars.DataFrame` objects.

    This function only checks differences in types on columns that are present in both dataframes
    (same column names, see :func:`get_column_name_differences` for more details).

    Currently the function is capable of cathing the following differences:
    - Different column types (e.g. `pl.Int64` vs `pl.Float64`)
    - Different time zones of same column type (e.g. `pl.Datetime(time_zone="UTC")` vs `pl.Datetime(time_zone="Europe/Berlin")`)
    - Different time precisions of same column type (e.g. `pl.Datetime(time_unit="ms")` vs `pl.Datetime(time_unit="us")`)

    Example:
        ```python
        import polars as pl
        from data_compare.src.comparator import get_column_dtype_differences

        df0 = pl.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [4, 5, 6],
            }
        )
        df1 = pl.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [4.0, 5.0, 6.0],
            }
        )

        same_columns, column_differences = get_column_dtype_differences(df0, df1, "df0", "df1")
        print(same_columns)
        print(column_differences)
        ```
        Output:
        ```
        ['a']

        [ColumnDifference(source='df0', column_name='b',
        difference_type=<ColumnDataTypeDifferenceType.DIFFERENT_TYPE: 'DIFFERENT_TYPE'>,
        more_information={'df0': Int64, 'df1': Float64})]
        ```


    .. note::
        **One thing to remember is that timezone of `pl.Datetime` column is defined by first value in the column.**

    Args:
        df0 (pl.DataFrame): The first dataframe.
        df1 (pl.DataFrame): The second dataframe.
        df0_name (str): The name of the first dataframe.
        df1_name (str): The name of the second dataframe.

    Returns:
        list[str]: The names of the columns that have the same type in both dataframes.

        list[:class:`data_compare.src.models.ColumnDifference`]: The differences in column types between the two dataframes.

    """
    df0_dtypes: dict[str, str] = {
        column_name: type(dtype) for column_name, dtype in zip(df0.columns, df0.dtypes)
    }
    df1_dtypes: dict[str, str] = {
        column_name: type(dtype) for column_name, dtype in zip(df1.columns, df1.dtypes)
    }

    same_columns, column_differences = get_column_name_differences(
        df0, df1, df0_name, df1_name
    )

    same_columns_after_dtpe_check: list[str] = []
    same_columns_after_dtpe_check.extend(same_columns)

    for same_col in same_columns:
        if df0_dtypes[same_col] != df1_dtypes[same_col]:
            column_differences.append(
                ColumnDifference(
                    source=df0_name,
                    column_name=same_col,
                    difference_type=ColumnDataTypeDifferenceType.DIFFERENT_TYPE,
                    more_information={
                        df0_name: df0_dtypes[same_col],
                        df1_name: df1_dtypes[same_col],
                    },
                )
            )
            same_columns_after_dtpe_check.remove(same_col)
            continue

        # check if the timezone is the same
        # in polars.DataFrame first element of the column is default timezone for the column
        if df0_dtypes[same_col] == pl.Datetime:
            if df0[same_col][0].tzinfo != df1[same_col][0].tzinfo:
                column_differences.append(
                    ColumnDifference(
                        source=df0_name,
                        column_name=same_col,
                        difference_type=ColumnDataTypeDifferenceType.DIFFERENT_TIMEZONE,
                        more_information={
                            df0_name: df0[same_col][0].tzinfo,
                            df1_name: df1[same_col][0].tzinfo,
                        },
                    )
                )
                same_columns_after_dtpe_check.remove(same_col)
                continue

            # check if the precision of time is the same
            if df0[same_col].dtype.time_unit != df1[same_col].dtype.time_unit:
                column_differences.append(
                    ColumnDifference(
                        source=df0_name,
                        column_name=same_col,
                        difference_type=ColumnDataTypeDifferenceType.DIFFERENT_TIME_PRECISION,
                        more_information={
                            df0_name: df0[same_col].dtype.time_unit,
                            df1_name: df1[same_col].dtype.time_unit,
                        },
                    )
                )
                same_columns_after_dtpe_check.remove(same_col)
                continue

    return same_columns_after_dtpe_check, column_differences


@convert_to_polars
@check_inputs
def get_row_differences(
    df0: pl.DataFrame,
    df1: pl.DataFrame,
    df0_name: str,
    df1_name: str,
) -> tuple[list[str], list[ColumnDifference], list[RowDifference]]:
    """
    Get the row differences between two dataframes, meaning find the rows that are in one dataframe but not in the other **or they differ**.


    This function compares only the rows that have the same columns and data types (:func:`get_column_dtype_differences`).


    Good thing is that this function also detects the duplicate rows, and also rows that are duplicated by a different number of times in each dataframe.

    Example:
        ```python
        import polars as pl
        from data_compare.src.comparator import get_row_differences

        df0 = pl.DataFrame(
            {
                "a": [1, 2, 3, 4],
                "b": [4, 5, 6, 7],
            }
        )
        df1 = pl.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [4.0, 5, 6.0],
            }
        )

        same_columns, column_differences, row_differences = get_row_differences(df0, df1, "df0", "df1")
        print(same_columns)
        print(column_differences)
        print(row_differences)
        ```
        Output:
        ```
        ['a']

        [ColumnDifference(source='df0', column_name='b',
        difference_type=<ColumnDataTypeDifferenceType.DIFFERENT_TYPE: 'DIFFERENT_TYPE'>,
        more_information={'df0': Int64, 'df1': Float64})]

        [RowDifference(source='df0',
        row={'a': [4]}, number_of_occurrences=1,
        difference_type=<RowDifferenceType.MISSING_ROW: 'MISSING_ROW'>,
        more_information=None)]
        ```

    Args:
        df0 (pl.DataFrame): The first dataframe.
        df1 (pl.DataFrame): The second dataframe.
        df0_name (str): The name of the first dataframe.
        df1_name (str): The name of the second dataframe.

    Returns:
       list[str]: The columns that are the same

       list[:class:`data_compare.src.models.ColumnDifference`]: The column differences (for more info see :func:`get_column_dtype_differences`)

       list[:class:`data_compare.src.models.RowDifference`]: The row differences.


    """
    same_columns, column_differences = get_column_dtype_differences(
        df0, df1, df0_name, df1_name
    )

    df0_subset: pl.DataFrame = df0.select(same_columns)
    df1_subset: pl.DataFrame = df1.select(same_columns)

    df0_subset = df0_subset.with_columns(df0_subset.hash_rows().alias("hash"))
    df1_subset = df1_subset.with_columns(df1_subset.hash_rows().alias("hash"))

    df0_subset = df0_subset.with_columns(pl.lit(df0_name).alias("source"))
    df1_subset = df1_subset.with_columns(pl.lit(df1_name).alias("source"))

    row_differences: list[RowDifference] = []

    differences_hash_df0: set[str] = set(df0_subset["hash"]).difference(
        set(df1_subset["hash"])
    )
    differences_hash_df1: set[str] = set(df1_subset["hash"]).difference(
        set(df0_subset["hash"])
    )

    for difference_hash in differences_hash_df0:
        difference_row: pl.DataFrame = df0_subset.filter(
            pl.col("hash") == difference_hash
        )

        diff: RowDifference = RowDifference(
            source=df0_name,
            row=difference_row.select(sorted(difference_row.columns))
            .drop(["hash", "source"])
            .sort("*")
            .to_dict(as_series=False),
            number_of_occurrences=difference_row.shape[0],
            difference_type=RowDifferenceType.MISSING_ROW,
        )
        row_differences.append(diff)

    for difference_hash in differences_hash_df1:
        difference_row: pl.DataFrame = df1_subset.filter(
            pl.col("hash") == difference_hash
        )

        diff: RowDifference = RowDifference(
            source=df1_name,
            row=difference_row.select(sorted(difference_row.columns))
            .drop(["hash", "source"])
            .sort("*")
            .to_dict(as_series=False),
            number_of_occurrences=difference_row.shape[0],
            difference_type=RowDifferenceType.MISSING_ROW,
        )
        row_differences.append(diff)

    # look for duplicates that are in both dataframes
    # but not the same number of times
    duplicates_df0 = df0_subset.filter(pl.col("hash").is_duplicated())
    duplicates_df1 = df1_subset.filter(pl.col("hash").is_duplicated())

    duplicates_df0_hashes: set[str] = set(duplicates_df0["hash"]) & set(
        df1_subset["hash"]
    )
    duplicates_df1_hashes: set[str] = set(duplicates_df1["hash"]) & set(
        df0_subset["hash"]
    )

    same_hashes: set[str] = duplicates_df0_hashes.union(duplicates_df1_hashes)
    for same_hash in same_hashes:
        duplicates_df0_count = df0_subset.filter(pl.col("hash") == same_hash).shape[0]
        duplicates_df1_count = df1_subset.filter(pl.col("hash") == same_hash).shape[0]

        if duplicates_df0_count == duplicates_df1_count:
            continue

        if duplicates_df0_count != duplicates_df1_count:
            if duplicates_df0_count > duplicates_df1_count:
                diff: RowDifference = RowDifference(
                    source=df0_name,
                    row=df0_subset.filter(pl.col("hash") == same_hash)
                    .select(sorted(df0_subset.columns))
                    .drop(["hash", "source"])
                    .sort("*")
                    .head(duplicates_df0_count - duplicates_df1_count)
                    .to_dict(as_series=False),
                    number_of_occurrences=duplicates_df0_count - duplicates_df1_count,
                    difference_type=RowDifferenceType.MISSING_ROW,
                )
                row_differences.append(diff)
                continue

            diff: RowDifference = RowDifference(
                source=df1_name,
                row=df1_subset.filter(pl.col("hash") == same_hash)
                .select(sorted(df1_subset.columns))
                .drop(["hash", "source"])
                .sort("*")
                .head(duplicates_df1_count - duplicates_df0_count)
                .to_dict(as_series=False),
                number_of_occurrences=duplicates_df1_count - duplicates_df0_count,
                difference_type=RowDifferenceType.MISSING_ROW,
            )
            row_differences.append(diff)

    return same_columns, column_differences, row_differences


def compare_group_column_by_column(
    data: pl.DataFrame, grouping_columns: list[str]
) -> list[Union[RowDifference, RowGroupDifference]]:
    """
    Compares the rows of a dataframe (**already**) grouped by the grouping columns.


    The main idea of this function is to recieve a dataframe that is grouped
    by the grouping columns and has a "source" column that indicates the source of the row.
    The function will then compare the rows of the dataframe and return the differences between the rows of the different sources.

    This function will return a list of :class:`data_compare.src.models.RowDifference` or :class:`data_compare.src.models.RowGroupDifference` objects:
    - The :class:`data_compare.src.models.RowDifference` object will be returned when there are only one source present in the group (*meaning that the row is only present in one source and not in the other*)
    - The :class:`data_compare.src.models.RowGroupDifference` object will be returned when there are multiple sources present in the group (*meaning that the row is present in multiple sources and the differences between the rows of the different sources are returned*).

    Example:
        ```python
        import polars as pl
        from data_compare.src.comparator import _get_row_differences_paired

        df0 = pl.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
        df1 = pl.DataFrame({"a": [1, 2, 3], "b": [1, 2, 10]})
        df0_name = "df0"
        df1_name = "df1"

        same_columns, different_columns, row_differences = _get_row_differences_paired(
            df0, df1, df0_name, df1_name, ["a"]
        )
        print(row_differences)
        ```
        Output:
        ```
        [RowGroupDifference(sources=['df0', 'df1'],
        row={'a': [3, 3], 'b': [3, 10]},
        number_of_occurrences=2,
        grouping_columns=['a'],
        column_differences=['b'],
        consise_information={'a': [3, 3], 'b': [3, 10], 'source': ['df0', 'df1']},
        row_with_source={'a': [3, 3], 'b': [3, 10], 'source': ['df0', 'df1']})]
        ```

    Raises:
        ValueError: If the dataframe is not grouped by the grouping columns.

    Args:
        data (pl.DataFrame): The dataframe to compare.
        grouping_columns (list[str]): The columns to group by.

    Returns:
        list[Union[:class:`data_compare.src.models.RowDifference`, :class:`data_compare.src.models.RowGroupDifference`]]: The differences between the rows of the different sources.


    """
    if data.select(grouping_columns).unique().shape[0] != 1:
        raise ValueError("The dataframe must be grouped by the grouping columns.")

    sources: list[str] = list(data["source"].unique())

    if len(sources) == 1:
        row_difference_information: RowDifference = RowDifference(
            source=sources[0],
            row=data.select(sorted(data.columns))
            .drop(["source"])
            .sort("*")
            .to_dict(as_series=False),
            number_of_occurrences=len(data),
            difference_type=RowDifferenceType.MISSING_ROW,
        )
        return row_difference_information

    different_columns: list[str] = []
    to_check_columns: set[str] = (
        set(data.columns) - set(grouping_columns) - {"hash", "source"}
    )
    for col in to_check_columns:
        if len(data[col].unique()) == 1:
            continue

        different_columns.append(col)

    row_grouping_difference: RowGroupDifference = RowGroupDifference(
        sources=sorted(sources),
        row=data.select(sorted(data.columns))
        .drop(["source"])
        .sort("*")
        .to_dict(as_series=False),
        number_of_occurrences=len(data),
        grouping_columns=sorted(grouping_columns),
        column_differences=sorted(different_columns),
        consise_information=data.select(
            sorted(grouping_columns + different_columns + ["source"])
        )
        .sort("*")
        .to_dict(as_series=False),
        row_with_source=data.select(sorted(data.columns))
        .sort("*")
        .to_dict(as_series=False),
    )
    return row_grouping_difference


@convert_to_polars
@check_inputs
def get_row_differences_paired(
    df0: pl.DataFrame,
    df1: pl.DataFrame,
    df0_name: str,
    df_1_name: str,
    grouping_columns: list[str],
) -> tuple[
    list[str], list[ColumnDifference], list[Union[RowDifference, RowGroupDifference]]
]:
    """
    Compares the rows of a dataframe grouped by the grouping columns.

    If you have two dataframes where some columns (or multiple of them) are identifying a row,
    with this function you can find the columns in which they differ, but also have a functionality that
    tells you that there are some missing rows in one of the dataframes.

    Example:
        ```python
        import polars as pl
        from data_compare.src.comparator import get_row_differences_paired
        df0 = pl.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
        df1 = pl.DataFrame({"a": [1, 2, 3], "b": [1, 2, 10]})
        df0_name = "df0"
        df1_name = "df1"
        same_columns, different_columns, row_differences = get_row_differences_paired(
            df0, df1, df0_name, df1_name, ["a"]
        )
        print(row_differences)
        ```
        Output:
        ```
        [RowGroupDifference(sources=['df0', 'df1'],
        row={'a': [3, 3], 'b': [3, 10]},
        number_of_occurrences=2,
        grouping_columns=['a'],
        column_differences=['b'],
        consise_information={'a': [3, 3], 'b': [3, 10], 'source': ['df0', 'df1']},
        row_with_source={'a': [3, 3], 'b': [3, 10], 'source': ['df0', 'df1']})]
        ```

    Raises:
        ValueError: If the pairing columns are not the present in both dataframes.

    Args:
        df0 (pl.DataFrame): The first dataframe.
        df1 (pl.DataFrame): The second dataframe.
        df0_name (str): The name of the first dataframe.
        df_1_name (str): The name of the second dataframe.
        grouping_columns (list[str]): The columns to group by.

    Returns:
        list[str]: The same columns

        list[:class:`data_compare.src.models.ColumnDifference`]: The column differences

        list[Union[:class:`data_compare.src.models.RowDifference`, :class:`data_compare.src.models.RowGroupDifference`]]: The row differences
    """
    same_columns, column_differences, row_differences = get_row_differences(
        df0, df1, df0_name, df_1_name
    )

    if len(set(grouping_columns).difference(same_columns)) > 0:
        raise ValueError(
            "Pairing columns must be the same in both dataframes. "
            f"Pairing columns: {grouping_columns}. Same columns: {same_columns}"
        )

    difference_dataframe: pl.DataFrame = convert_row_differences_to_pandas(
        row_differences
    )
    if len(difference_dataframe) == 0:
        return same_columns, column_differences, row_differences

    row_differences: list[Union[RowDifference, RowGroupDifference]] = []
    for name, dat in difference_dataframe.group_by(grouping_columns):
        difference: Union[RowDifference, RowGroupDifference] = (
            compare_group_column_by_column(dat, grouping_columns)
        )
        row_differences.append(difference)
    return same_columns, column_differences, row_differences


@convert_to_polars
@check_inputs
def get_data_report(
    df0: pl.DataFrame,
    df1: pl.DataFrame,
    df0_name: str,
    df1_name: str,
    grouping_columns: Optional[list[str]] = None,
) -> DataReport:
    """
    Get a data report comparing two dataframes.

    This function returns multiple informations like:
    - the length of the dataframes
    - the columns that are the same in both dataframes (have same column name and data type)
    - the columns that are different in both dataframes (they have different data types or not in both dataframes)
    - the rows that are different in both dataframes (they are not existing or they have different column values if grouping parameter is used)
    - total number of row differences (on columns that are same in both dataframes)
    - total number of row differences in the first dataframe
    - total number of row differences in the second dataframe
    - the ratio of row differences in the first dataframe
    - the ratio of row differences in the second dataframe

    .. note::
       The ratio of row differences is calculated as the
       number of row differences from some of the dataframes divided by the total number of row differences.

    Example:
        ```python
        import polars as pl
        from data_compare.src.comparator import get_data_report
        df0 = pl.DataFrame({"a": [1, 2, 3, 3, 3, 4], "b": [1, 2, 3, 10, 10, 15]})
        df1 = pl.DataFrame({"a": [1, 2, 3, 3, 4, 5], "b": [1, 2, 3, 10, 20, 24]})
        df0_name = "df0"
        df1_name = "df1"
        report = get_data_report(df0, df1, df0_name, df1_name, ["a"])
        print(report.model_dump_json(indent=4))
        ```

        Output:
        ```
        {
            "df0_length": 6,
            "df1_length": 6,
            "df0_name": "df0",
            "df1_name": "df1",
            "comparable_columns": [
                "b",
                "a"
            ],
            "column_differences": [],
            "row_differences": [
                {
                    "source": "df1",
                    "row": {
                        "a": [
                            5
                        ],
                        "b": [
                            24
                        ]
                    },
                    "number_of_occurrences": 1,
                    "difference_type": "MISSING_ROW",
                    "more_information": null
                },
                {
                    "source": "df0",
                    "row": {
                        "a": [
                            3
                        ],
                        "b": [
                            10
                        ]
                    },
                    "number_of_occurrences": 1,
                    "difference_type": "MISSING_ROW",
                    "more_information": null
                },
                {
                    "sources": [
                        "df0",
                        "df1"
                    ],
                    "row": {
                        "a": [
                            4,
                            4
                        ],
                        "b": [
                            15,
                            20
                        ]
                    },
                    "number_of_occurrences": 2,
                    "grouping_columns": [
                        "a"
                    ],
                    "column_differences": [
                        "b"
                    ],
                    "consise_information": {
                        "a": [
                            4,
                            4
                        ],
                        "b": [
                            15,
                            20
                        ],
                        "source": [
                            "df0",
                            "df1"
                        ]
                    },
                    "row_with_source": {
                        "a": [
                            4,
                            4
                        ],
                        "b": [
                            15,
                            20
                        ],
                        "source": [
                            "df0",
                            "df1"
                        ]
                    }
                }
            ],
            "number_of_row_differences": 4,
            "number_of_differences_source_0": 2,
            "number_of_differences_source_1": 2,
            "ratio_of_difference_from_source_0": 0.5,
            "ratio_of_difference_from_source_1": 0.5
        }
        ```

    Args:
        df0 (pl.DataFrame): The first dataframe.
        df1 (pl.DataFrame): The second dataframe.

    Returns:
        :class:`data_compare.src.models.DataReport`: A data report comparing the two dataframes.
    """
    if grouping_columns is None:
        same_columns, column_differences, row_differences = get_row_differences(
            df0, df1, df0_name, df1_name
        )
    else:
        same_columns, column_differences, row_differences = get_row_differences_paired(
            df0, df1, df0_name, df1_name, grouping_columns
        )
    return DataReport(
        df0_length=len(df0),
        df1_length=len(df1),
        df1=df1,
        df0_name=df0_name,
        df1_name=df1_name,
        comparable_columns=same_columns,
        row_differences=row_differences,
        column_differences=column_differences,
    )
