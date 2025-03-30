from typing import Any, Optional, Union

from pydantic import BaseModel, computed_field
import polars as pl

from data_compare.src.difference_types import (
    ColumnNameDifferenceType,
    ColumnDataTypeDifferenceType,
    RowDifferenceType,
)


class RowDifference(BaseModel):
    source: str
    row: dict[str, Any]
    number_of_occurrences: int
    difference_type: RowDifferenceType
    more_information: Optional[Any] = None

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
    sources: list[str]
    row: dict[str, Any]
    number_of_occurrences: int
    grouping_columns: list[str]
    column_differences: list[str]
    consise_information: dict[str, Any]
    row_with_source: dict[str, Any]

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
    source: str
    column_name: str
    difference_type: Union[ColumnNameDifferenceType, ColumnDataTypeDifferenceType]
    more_information: Optional[Any] = None

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
    df0_length: int
    df1_length: int
    df0_name: str
    df1_name: str

    comparable_columns: list[str]
    column_differences: list[ColumnDifference]
    row_differences: list[Union[RowDifference, RowGroupDifference]]

    @computed_field
    @property
    def number_of_row_differences(self) -> int:
        return sum([rd.number_of_occurrences for rd in self.row_differences])

    @computed_field
    @property
    def number_of_differences_source_0(self) -> int:
        counter: int = 0
        for rd in self.row_differences:
            if isinstance(rd, RowDifference) and rd.source == self.df1_name:
                continue

            if isinstance(rd, RowDifference) and rd.source == self.df0_name:
                counter += rd.number_of_occurrences
                continue

            counter += sum(
                [1 for x in rd.consise_information["source"] if x == self.df0_name]
            )

        return counter

    @computed_field
    @property
    def number_of_differences_source_1(self) -> int:
        counter: int = 0
        for rd in self.row_differences:
            if isinstance(rd, RowDifference) and rd.source == self.df0_name:
                continue

            if isinstance(rd, RowDifference) and rd.source == self.df1_name:
                counter += rd.number_of_occurrences
                continue

            counter += sum(
                [1 for x in rd.consise_information["source"] if x == self.df1_name]
            )

        return counter

    @computed_field
    @property
    def ratio_of_difference_from_source_0(self) -> float:
        if self.number_of_row_differences == 0:
            return 0.0

        return self.number_of_differences_source_0 / self.number_of_row_differences

    @computed_field
    @property
    def ratio_of_difference_from_source_1(self) -> float:
        if self.number_of_row_differences == 0:
            return 0.0

        return self.number_of_differences_source_1 / self.number_of_row_differences
