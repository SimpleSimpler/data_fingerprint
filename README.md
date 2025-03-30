<p align="center">
  <img src="items/SimpleSimpler_logo_green.svg" />
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
import pandas as pd
from data_fingerprint import get_data_report

# Create two sample datasets
df1 = pd.DataFrame({
    'id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35]
})
df2 = pd.DataFrame({
    'id': [1, 2, 4],
    'name': ['Alice', 'Bob', 'David'],
    'age': [25, 30, 40]
})
# Generate a data report comparing the two datasets
report = get_data_report(df1, df2, 'df1', 'df2', 'id')
print(report.model_dump_json(indent=4))

# get differences in pandas.DataFrame format

df = report.to_dataframe()
print(df)
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## Contact

For any questions or feedback, please contact [your email].