<p align="center">
  <img src="https://imgur.com/EvSWq14.png" />
</p>

# DataFingerprint

**DataFingerprint** is a Python package designed to compare two datasets and generate a detailed report highlighting the differences between them. This tool is particularly useful for data validation, quality assurance, and ensuring data consistency across different sources.

## Features

- **Column Name Differences**: Identify columns that are present in one dataset but missing in the other.
- **Column Data Type Differences**: Detect discrepancies in data types between corresponding columns in the two datasets.
- **Row Differences**: Find rows that are present in one dataset but missing in the other, or rows that have different values in corresponding columns.
- **Paired Row Differences**: Compare rows that have the same primary key or unique identifier in both datasets and identify differences in their values.
- **Data Report**: Generate a comprehensive report summarizing all the differences found between the two datasets.

## Installation

To install DataFingerprint, you can use pip:
```bash
pip install data-fingerprint
```

## Usage

Here's a basic example of how to use DataFingerprint to compare two datasets:
```python
import polars as pl

from data_fingerprint.src.utils import get_dataframe
from data_fingerprint.src.comparator import get_data_report
from data_fingerprint.src.models import DataReport

# Create two sample datasets
df1 = pl.DataFrame(
    {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]}
)
df2 = pl.DataFrame(
    {"id": [1, 2, 4], "name": ["Alice", "Bob", "David"], "age": [25, 30, 40]}
)
# Generate a data report comparing the two datasets
report: DataReport = get_data_report(df1, df2, "df_0", "df_1", grouping_columns=["id"])
print(report.model_dump_json(indent=4))
print(get_dataframe(report))
```

## License

This project is licensed under the GPLv3 License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## Contact

For any questions or feedback, please contact [your email].