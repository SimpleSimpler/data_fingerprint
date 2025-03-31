import datetime

import pytest
import polars as pl
import pandas as pd
from zoneinfo import ZoneInfo

from data_fingerprint.src.comparator import (
    get_column_name_differences,
    get_column_dtype_differences,
    get_row_differences,
    get_row_differences_paired,
    get_data_report,
)
from data_fingerprint.src.models import (
    ColumnDifference,
    RowDifference,
    RowGroupDifference,
)
from data_fingerprint.src.difference_types import (
    ColumnNameDifferenceType,
    ColumnDataTypeDifferenceType,
    RowDifferenceType,
)

from data_fingerprint.src.utils import (
    get_dataframe,
    get_number_of_row_differences,
    get_number_of_differences_per_source,
    get_ratio_of_differences_per_source,
    get_column_difference_ratio,
)


def test_get_column_name_differences():
    df0 = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    df1 = pl.DataFrame({"a": [1, 2], "c": [3, 4]})
    df0_name = "df0"
    df1_name = "df1"
    expected_differences = set(
        [
            ColumnDifference(
                source="df0",
                column_name="c",
                difference_type=ColumnNameDifferenceType.MISSING,
            ),
            ColumnDifference(
                source="df0",
                column_name="b",
                difference_type=ColumnNameDifferenceType.EXTRA,
            ),
        ]
    )
    same_columns, different_columns = get_column_name_differences(
        df0, df1, df0_name, df1_name
    )
    assert set(different_columns) == expected_differences
    assert set(same_columns) == {"a"}


def test_get_column_name_differences_no_differences():
    df0 = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    df1 = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    df0_name = "df0"
    df1_name = "df1"
    expected_differences = set()
    same_columns, different_columns = get_column_name_differences(
        df0, df1, df0_name, df1_name
    )
    assert set(different_columns) == expected_differences
    assert set(same_columns) == {"a", "b"}


def test_get_column_dtype_differences():
    df0 = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    df1 = pl.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    df0_name = "df0"
    df1_name = "df1"
    expected_differences = set(
        [
            ColumnDifference(
                source="df0",
                column_name="b",
                difference_type=ColumnNameDifferenceType.DIFFERENT_TYPE,
                more_information={"df0": pl.Int64, "df1": pl.Float64},
            )
        ]
    )
    same_columns, different_columns = get_column_dtype_differences(
        df0, df1, df0_name, df1_name
    )
    assert set(different_columns) == expected_differences
    assert set(same_columns) == {"a"}


def test_get_column_dtype_differences_timezone():
    df0 = pl.DataFrame(
        {
            "a": [1, 2],
            "b": [
                datetime.datetime(2021, 1, 1, tzinfo=datetime.timezone.utc),
                datetime.datetime(2021, 1, 2),
            ],
        }
    )
    df1 = pl.DataFrame(
        {
            "a": [1, 2],
            "b": [datetime.datetime(2021, 1, 1), datetime.datetime(2021, 1, 2)],
            "c": [1, 2],
        }
    )
    df0_name = "df0"
    df1_name = "df1"
    expected_differences = set(
        [
            ColumnDifference(
                source="df0",
                column_name="b",
                difference_type=ColumnDataTypeDifferenceType.DIFFERENT_TIMEZONE,
                more_information={"df0": ZoneInfo("UTC"), "df1": None},
            ),
            ColumnDifference(
                source="df0",
                column_name="c",
                difference_type=ColumnNameDifferenceType.MISSING,
            ),
        ]
    )
    same_columns, different_columns = get_column_dtype_differences(
        df0, df1, df0_name, df1_name
    )
    assert set(different_columns) == expected_differences
    assert set(same_columns) == {"a"}


def test_get_column_dtype_differences_time_precision():
    df0 = pd.DataFrame(
        {
            "a": [1, 2],
            "b": [
                datetime.datetime(2021, 1, 1),
                datetime.datetime(2021, 1, 2),
            ],
        }
    )
    df1 = pl.DataFrame(
        {
            "a": [1, 2],
            "b": [datetime.datetime(2021, 1, 1), datetime.datetime(2021, 1, 2)],
            "c": [1, 2],
        }
    )
    df0_name = "df0"
    df1_name = "df1"
    expected_differences = set(
        [
            ColumnDifference(
                source="df0",
                column_name="b",
                difference_type=ColumnDataTypeDifferenceType.DIFFERENT_TIME_PRECISION,
                more_information={"df0": "ns", "df1": "us"},
            ),
            ColumnDifference(
                source="df0",
                column_name="c",
                difference_type=ColumnNameDifferenceType.MISSING,
            ),
        ]
    )
    with pytest.warns(UserWarning, match=".*Trasnforming pandas DataFrames.*"):
        same_columns, different_columns = get_column_dtype_differences(
            df0, df1, df0_name, df1_name
        )
    assert set(different_columns) == expected_differences
    assert set(same_columns) == {"a"}


def test_get_row_differences_no_differences():
    df0 = pl.DataFrame(
        {
            "a": [1, 2],
            "b": [datetime.datetime(2021, 1, 1), datetime.datetime(2021, 1, 2)],
        }
    )
    df1 = pl.DataFrame(
        {
            "a": [1, 2],
            "b": [datetime.datetime(2021, 1, 1), datetime.datetime(2021, 1, 2)],
        }
    )
    df0_name = "df0"
    df1_name = "df1"
    expected_differences = []
    same_columns, different_columns, row_differences = get_row_differences(
        df0, df1, df0_name, df1_name
    )
    assert set(different_columns) == set(expected_differences)
    assert set(same_columns) == {"a", "b"}
    assert set(row_differences) == set([])


def test_get_row_differences_with_differences():
    df0 = pl.DataFrame(
        {
            "a": [1, 2],
            "b": [datetime.datetime(2021, 1, 1), datetime.datetime(2021, 1, 2)],
        }
    )
    df1 = pl.DataFrame(
        {
            "a": [1, 2],
            "b": [datetime.datetime(2021, 1, 1), datetime.datetime(2021, 1, 3)],
        }
    )
    df0_name = "df0"
    df1_name = "df1"
    expected_row_differences = [
        RowDifference(
            source="df1",
            row={"a": [2], "b": [datetime.datetime(2021, 1, 3)]},
            number_of_occurrences=1,
            difference_type=RowDifferenceType.MISSING_ROW,
        ),
        RowDifference(
            source="df0",
            row={"a": [2], "b": [datetime.datetime(2021, 1, 2)]},
            number_of_occurrences=1,
            difference_type=RowDifferenceType.MISSING_ROW,
        ),
    ]
    expected_column_differences = []
    same_columns, different_columns, row_differences = get_row_differences(
        df0, df1, df0_name, df1_name
    )
    assert set(different_columns) == set(expected_column_differences)
    assert set(same_columns) == {"a", "b"}
    assert set(row_differences) == set(expected_row_differences)


def test_get_row_differences_with_differences_duplicates():
    df0 = pl.DataFrame(
        {
            "a": [1, 2, 2],
            "b": [
                datetime.datetime(2021, 1, 1),
                datetime.datetime(2021, 1, 2),
                datetime.datetime(2021, 1, 2),
            ],
        }
    )
    df1 = pl.DataFrame(
        {
            "a": [1, 2],
            "b": [datetime.datetime(2021, 1, 1), datetime.datetime(2021, 1, 3)],
        }
    )
    df0_name = "df0"
    df1_name = "df1"
    expected_row_differences = [
        RowDifference(
            source="df1",
            row={"a": [2], "b": [datetime.datetime(2021, 1, 3)]},
            number_of_occurrences=1,
            difference_type=RowDifferenceType.MISSING_ROW,
        ),
        RowDifference(
            source="df0",
            row={
                "a": [2, 2],
                "b": [datetime.datetime(2021, 1, 2), datetime.datetime(2021, 1, 2)],
            },
            number_of_occurrences=2,
            difference_type=RowDifferenceType.MISSING_ROW,
        ),
    ]
    expected_column_differences = []
    same_columns, different_columns, row_differences = get_row_differences(
        df0, df1, df0_name, df1_name
    )
    assert set(different_columns) == set(expected_column_differences)
    assert set(same_columns) == {"a", "b"}
    assert set(row_differences) == set(expected_row_differences)


def test_get_row_differences_with_differences_duplicates_multiple():
    df0 = pl.DataFrame(
        {
            "a": [1, 2, 2, 2, 2],
            "b": [
                datetime.datetime(2021, 1, 1),
                datetime.datetime(2021, 1, 2),
                datetime.datetime(2021, 1, 2),
                datetime.datetime(2021, 1, 2),
                datetime.datetime(2021, 1, 2),
            ],
        }
    )
    df1 = pl.DataFrame(
        {
            "a": [1, 2, 2],
            "b": [
                datetime.datetime(2021, 1, 1),
                datetime.datetime(2021, 1, 3),
                datetime.datetime(2021, 1, 2),
            ],
        }
    )
    df0_name = "df0"
    df1_name = "df1"
    expected_row_differences = [
        RowDifference(
            source="df1",
            row={"a": [2], "b": [datetime.datetime(2021, 1, 3)]},
            number_of_occurrences=1,
            difference_type=RowDifferenceType.MISSING_ROW,
        ),
        RowDifference(
            source="df0",
            row={
                "a": [2, 2, 2],
                "b": [
                    datetime.datetime(2021, 1, 2),
                    datetime.datetime(2021, 1, 2),
                    datetime.datetime(2021, 1, 2),
                ],
            },
            number_of_occurrences=3,
            difference_type=RowDifferenceType.MISSING_ROW,
        ),
    ]
    expected_column_differences = []
    same_columns, different_columns, row_differences = get_row_differences(
        df0, df1, df0_name, df1_name
    )
    print(row_differences)
    assert set(different_columns) == set(expected_column_differences)
    assert set(same_columns) == {"a", "b"}
    assert set(row_differences) == set(expected_row_differences)


def test_grouping_row_difference():
    df0 = pl.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    df1 = pl.DataFrame({"a": [1, 2, 3], "b": [1, 2, 10]})
    df0_name = "df0"
    df1_name = "df1"
    expected_row_differences = [
        RowGroupDifference(
            sources=["df0", "df1"],
            row={"a": [3, 3], "b": [3, 10]},
            number_of_occurrences=2,
            grouping_columns=["a"],
            column_differences=["b"],
            consise_information={"a": [3, 3], "b": [3, 10], "source": ["df0", "df1"]},
            row_with_source={"a": [3, 3], "b": [3, 10], "source": ["df0", "df1"]},
        )
    ]
    same_columns, different_columns, row_differences = get_row_differences_paired(
        df0, df1, df0_name, df1_name, ["a"]
    )
    print(row_differences)
    assert set(different_columns) == set([])
    assert set(same_columns) == {"a", "b"}
    assert set(row_differences) == set(expected_row_differences)


def test_grouping_row_difference_multiple_same_duplicates():
    df0 = pl.DataFrame({"a": [1, 2, 3, 3], "b": [1, 2, 3, 10]})
    df1 = pl.DataFrame({"a": [1, 2, 3, 3], "b": [1, 2, 3, 10]})
    df0_name = "df0"
    df1_name = "df1"
    expected_row_differences = []
    same_columns, different_columns, row_differences = get_row_differences_paired(
        df0, df1, df0_name, df1_name, ["a"]
    )
    print(row_differences)
    assert set(different_columns) == set([])
    assert set(same_columns) == {"a", "b"}
    assert set(row_differences) == set(expected_row_differences)


def test_grouping_row_difference_multiple_duplicates():
    df0 = pl.DataFrame({"a": [1, 2, 3, 3, 3], "b": [1, 2, 3, 10, 10]})
    df1 = pl.DataFrame({"a": [1, 2, 3, 3], "b": [1, 2, 3, 10]})
    df0_name = "df0"
    df1_name = "df1"
    expected_row_differences = [
        RowDifference(
            row={"a": [3], "b": [10]},
            source=df0_name,
            difference_type=RowDifferenceType.MISSING_ROW,
            number_of_occurrences=1,
        )
    ]
    same_columns, different_columns, row_differences = get_row_differences_paired(
        df0, df1, df0_name, df1_name, ["a"]
    )
    print(row_differences)
    assert set(different_columns) == set([])
    assert set(same_columns) == {"a", "b"}
    assert set(row_differences) == set(expected_row_differences)


def test_data_report():
    df0 = pl.DataFrame({"a": [1, 2, 3, 3, 3, 4], "b": [1, 2, 3, 10, 10, 15]})
    df1 = pl.DataFrame({"a": [1, 2, 3, 3, 4, 5], "b": [1, 2, 3, 10, 20, 24]})
    df0_name = "df0"
    df1_name = "df1"
    report = get_data_report(df0, df1, df0_name, df1_name, ["a"])
    assert get_number_of_row_differences(report) == 4
    assert get_number_of_differences_per_source(report) == {df0_name: 2, df1_name: 2}
    assert get_ratio_of_differences_per_source(report) == {df0_name: 0.5, df1_name: 0.5}
    assert get_number_of_row_differences(report) == len(get_dataframe(report))


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
