from decimal import Decimal
import datetime
import time

import pandas as pd
import polars as pl

from data_fingerprint.src.comparator import get_data_report
from data_fingerprint.src.models import DataReport
from data_fingerprint.src.utils import get_dataframe

if __name__ == "__main__":
    df_0 = pl.read_csv("samples/simple_0.csv", has_header=True)
    df_1 = pd.read_csv("samples/simple_1.csv")

    report: DataReport = get_data_report(
        df_0, df_1, "df_0", "df_1", pairing_columns=["a"]
    )
    print(report.model_dump_json(indent=4))
    print(get_dataframe(report))
