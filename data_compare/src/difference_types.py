from enum import Enum


class ColumnNameDifferenceType(str, Enum):
    MISSING: str = "MISSING"
    EXTRA: str = "EXTRA"
    DIFFERENT_TYPE: str = "DIFFERENT_TYPE"


class ColumnDataTypeDifferenceType(str, Enum):
    DIFFERENT_TYPE: str = "DIFFERENT_TYPE"
    DIFFERENT_TIMEZONE: str = "DIFFERENT_TIMEZONE"
    DIFFERENT_TIME_PRECISION: str = "DIFFERENT_TIME_PRECISION"


class RowDifferenceType(str, Enum):
    MISSING_ROW: str = "MISSING_ROW"
