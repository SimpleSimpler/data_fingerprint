from enum import Enum


class ColumnNameDifferenceType(str, Enum):
    """
    Enum for column name difference types.
    """

    MISSING: str = "MISSING"
    """When this column is missing from the first source."""

    EXTRA: str = "EXTRA"
    """When this column is extra in the first source (**not** in the second source)."""

    DIFFERENT_TYPE: str = "DIFFERENT_TYPE"
    """When this column is present in both sources but has a different data type."""


class ColumnDataTypeDifferenceType(str, Enum):
    """
    Enum for column data type difference types.
    This is used when the column names are the same but the data types are different.
    """

    DIFFERENT_TYPE: str = "DIFFERENT_TYPE"
    """When this column is present in both sources but has a different data type."""

    DIFFERENT_TIMEZONE: str = "DIFFERENT_TIMEZONE"
    """When this column is present in both sources but has a different timezone."""

    DIFFERENT_TIME_PRECISION: str = "DIFFERENT_TIME_PRECISION"
    """When this column is present in both sources but has a different time precision."""


class RowDifferenceType(str, Enum):
    """
    Enum for row difference types.
    """

    MISSING_ROW: str = "MISSING_ROW"
    """When this row is present in `source` row but not in the other source."""
