from typing import Union, Optional

import polars as pl

from data_compare.src.models import (
    ColumnDifference,
    RowDifference,
    RowGroupDifference,
    DataReport,
)
from data_compare.src.utils import convert_to_polars, convert_row_differences_to_pandas
from data_compare.src.checkers import check_inputs
from data_compare.src.difference_types import (
    ColumnNameDifferenceType,
    ColumnDataTypeDifferenceType,
    RowDifferenceType,
)


@convert_to_polars
@check_inputs
def _get_column_name_differences(
    df0: pl.DataFrame, df1: pl.DataFrame, df0_name: str, df1_name: str
) -> tuple[list[str], list[ColumnDifference]]:
    """
    Get the differences in column names between two DataFrames.
    The referent DataFrame is df0. The differences are returned as a list of ColumnDifference objects.

    Args:
        df0 (pl.DataFrame): The first DataFrame.
        df1 (pl.DataFrame): The second DataFrame.
        df0_name (str): The name of the first DataFrame.
        df1_name (str): The name of the second DataFrame.

    Returns:
        list[str]: A list of column names that are the same in both DataFrames.
        list[ColumnDifference]: A list of ColumnDifference objects representing the differences in column names.
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
def _get_column_dtype_differences(
    df0: pl.DataFrame, df1: pl.DataFrame, df0_name: str, df1_name: str
) -> tuple[list[str], list[ColumnDifference]]:
    """
    Get the differences in column types between two dataframes.

    Args:
        df0 (pl.DataFrame): The first dataframe.
        df1 (pl.DataFrame): The second dataframe.
        df0_name (str): The name of the first dataframe.
        df1_name (str): The name of the second dataframe.

    Returns:
        list[str]: The names of the columns that have the same type in both dataframes.
        list[ColumnDifference]: The differences in column types between the two dataframes.

    """
    df0_dtypes: dict[str, str] = {
        column_name: type(dtype) for column_name, dtype in zip(df0.columns, df0.dtypes)
    }
    df1_dtypes: dict[str, str] = {
        column_name: type(dtype) for column_name, dtype in zip(df1.columns, df1.dtypes)
    }

    same_columns, column_differences = _get_column_name_differences(
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
def _get_row_differences(
    df0: pl.DataFrame,
    df1: pl.DataFrame,
    df0_name: str,
    df1_name: str,
) -> tuple[list[str], list[ColumnDifference], list[RowDifference]]:
    same_columns, column_differences = _get_column_dtype_differences(
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
    print("same_hashes", same_hashes)
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


def _compare_group_column_by_column(
    data: pl.DataFrame, grouping_columns: list[str]
) -> Union[RowDifference, RowGroupDifference]:
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
def _get_row_differences_paired(
    df0: pl.DataFrame,
    df1: pl.DataFrame,
    df0_name: str,
    df_1_name: str,
    pairing_columns: list[str],
) -> tuple[list[str], list[ColumnDifference], list[RowDifference]]:
    same_columns, column_differences, row_differences = _get_row_differences(
        df0, df1, df0_name, df_1_name
    )

    if len(set(pairing_columns).difference(same_columns)) > 0:
        raise ValueError(
            "Pairing columns must be the same in both dataframes. "
            f"Pairing columns: {pairing_columns}. Same columns: {same_columns}"
        )

    difference_dataframe: pl.DataFrame = convert_row_differences_to_pandas(
        row_differences
    )
    if len(difference_dataframe) == 0:
        return same_columns, column_differences, row_differences

    row_differences: list[Union[RowDifference, RowGroupDifference]] = []
    for name, dat in difference_dataframe.group_by(pairing_columns):
        difference: Union[RowDifference, RowGroupDifference] = (
            _compare_group_column_by_column(dat, pairing_columns)
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
    pairing_columns: Optional[list[str]] = None,
) -> DataReport:
    if pairing_columns is None:
        same_columns, column_differences, row_differences = _get_row_differences(
            df0, df1, df0_name, df1_name
        )
    else:
        same_columns, column_differences, row_differences = _get_row_differences_paired(
            df0, df1, df0_name, df1_name, pairing_columns
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
