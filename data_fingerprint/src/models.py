from typing import Any, Optional, Union

from pydantic import BaseModel, computed_field
import polars as pl

from data_fingerprint.src.difference_types import (
    ColumnNameDifferenceType,
    ColumnDataTypeDifferenceType,
    RowDifferenceType,
)


class RowDifference(BaseModel):
    """
    Model for row differences.
    """

    source: str
    """The data source name of the row."""

    row: dict[str, Any]
    """The **RAW** row that is different."""

    number_of_occurrences: int
    """The number of times this row is present in the source."""

    difference_type: RowDifferenceType
    """The type of difference."""

    more_information: Optional[Any] = None
    """More information about the difference."""

    def __hash__(self):
        return hash(
            (
                self.source,
                str(self.row),
                self.number_of_occurrences,
                self.difference_type,
                str(self.more_information),
            )
        )


class RowGroupDifference(BaseModel):
    """
    Model for row group differences.
    """

    sources: list[str]
    """The sources of the different rows."""

    row: dict[str, Any]
    """The **RAW** row that is different."""

    number_of_occurrences: int
    """The number of times this difference is present in all sources (**total**)."""

    grouping_columns: list[str]
    """The columns used to group the rows."""

    column_differences: list[str]
    """The columns that are different."""

    consise_information: dict[str, Any]
    """
    More information about the difference. 
    This view only shows the differences between the sources, 
    it is not showing the columns that are the same.
    """

    row_with_source: dict[str, Any]
    """
    The row with the source name as column.
    """

    def __hash__(self):
        return hash(
            (
                str(sorted(self.sources)),
                str(self.row),
                self.number_of_occurrences,
                str(sorted(self.grouping_columns)),
                str(sorted(self.column_differences)),
                str(self.consise_information),
            )
        )


class ColumnDifference(BaseModel):
    """
    Model for column differences.
    """

    source: str
    """The source of the column difference."""

    column_name: str
    """The name of the column that is different."""

    difference_type: Union[ColumnNameDifferenceType, ColumnDataTypeDifferenceType]
    """The type of difference."""

    more_information: Optional[Any] = None
    """More information about the difference."""

    def __hash__(self):
        return hash(
            (
                self.source,
                self.column_name,
                self.difference_type,
                str(self.more_information),
            )
        )


class DataReport(BaseModel):
    """
    Model for data report.
    """

    df0_length: int
    """The length of the first dataframe."""

    df1_length: int
    """The length of the second dataframe."""

    df0_name: str
    """The name of the first dataframe."""

    df1_name: str
    """The name of the second dataframe."""

    comparable_columns: list[str]
    """The columns that are comparable (*same name and same data type*)."""

    column_differences: list[ColumnDifference]
    """The column differences."""

    row_differences: list[Union[RowDifference, RowGroupDifference]]
    """The row differences."""
