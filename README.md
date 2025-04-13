<p align="center">
  <img src="https://imgur.com/EvSWq14.png" />
</p>

<p align="center">
  <a href="https://datafingerprinttry.streamlit.app">
    <img src="https://imgur.com/Wbsks2r.png" width="200"/>
  </a>
</p>

# DataFingerprint

**DataFingerprint** is a Python package designed to compare two datasets and generate a detailed report highlighting the differences between them. This tool is particularly useful for data validation, quality assurance, and ensuring data consistency across different sources.

# Try it out on streamlit
https://datafingerprinttry.streamlit.app

## Features

- **Column Name Differences**: Identify columns that are present in one dataset but missing in the other.
- **Column Data Type Differences**: Detect discrepancies in data types between corresponding columns in the two datasets.
- **Row Differences**: Find rows that are present in one dataset but missing in the other, or rows that have different values in corresponding columns.
- **Paired Row Differences**: Compare rows that have the same primary key or unique identifier in both datasets and identify differences in their values.
- **Data Report**: Generate a comprehensive report summarizing all the differences found between the two datasets.
- **Grouping Threshold**: When you use grouping you can specify a threhold for each column to ignore small differences between the same group.

| function                                                        | purpose                                                                   | result                                 |
|-----------------------------------------------------------------|---------------------------------------------------------------------------|----------------------------------------|
| `data_fingerprint.src.comparator.get_data_report`                 | Get data report object that has all the information about the differences | `data_fingerprint.src.models.DataReport` |
| `data_fingerprint.src.utils.get_dataframe`                        | Get polars.Dataframe of rows that are different (added source column)     | `polars.DataFrame`                       |
| `data_fingerprint.src.utils.get_number_of_row_differences`        | Get the number of different rows                                          | `int`                                    |
| `data_fingerprint.src.utils.get_number_of_differences_per_source` | Get the number of row differences per source                              | `dict[str, int]`                         |
| `data_fingerprint.src.utils.get_ratio_of_differences_per_source`  | Get the ratio of row differences per source                               | `dict[str, float]`                       |
| `data_fingerprint.src.utils.get_column_difference_ratio`          | [When grouping is used] Get the distribution of differences per column    | `dict[str, float]`                       |

## Installation

To install DataFingerprint, you can use pip:
```bash
pip install data-fingerprint
```

## Examples

Here's a basic example of how to use DataFingerprint to compare two datasets:
```python
import polars as pl

from data_fingerprint.src.utils import get_dataframe
from data_fingerprint.src.comparator import get_data_report
from data_fingerprint.src.models import DataReport

# Create two sample datasets
df0 = pl.DataFrame(
    {
        "id": [1, 2, 3, 4],
        "name": ["Alice", "Bob", "Charlie", "George"],
        "age": [25, 30, 35, 26],
        "height": [170, 180, 175, 160],
        "weight": [60, 70, 75, 65],
    }
)
df1 = pl.DataFrame(
    {
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "David"],
        "age": [25, 30, 35],
        "weight": ["60", "70", "75"],
        "married": [True, False, True],
    }
)
# Generate a data report comparing the two datasets
report: DataReport = get_data_report(df0, df1, "df_0", "df_1", grouping_columns=["id"])
print(report.model_dump_json(indent=4))
```
Output:
```json
{
    "df0_length": 4,
    "df1_length": 3,
    "df0_name": "df_0",
    "df1_name": "df_1",
    "comparable_columns": [
        "name",
        "id",
        "age"
    ],
    "column_differences": [
        {
            "source": "df_0",
            "column_name": "married",
            "difference_type": "MISSING",
            "more_information": null
        },
        {
            "source": "df_0",
            "column_name": "height",
            "difference_type": "EXTRA",
            "more_information": null
        },
        {
            "source": "df_0",
            "column_name": "weight",
            "difference_type": "DIFFERENT_TYPE",
            "more_information": {
                "df_0": "Int64",
                "df_1": "String"
            }
        }
    ],
    "row_differences": [
        {
            "source": "df_0",
            "row": {
                "age": [
                    26
                ],
                "id": [
                    4
                ],
                "name": [
                    "George"
                ]
            },
            "number_of_occurrences": 1,
            "difference_type": "MISSING_ROW",
            "more_information": null
        },
        {
            "sources": [
                "df_0",
                "df_1"
            ],
            "row": {
                "age": [
                    35,
                    35
                ],
                "id": [
                    3,
                    3
                ],
                "name": [
                    "Charlie",
                    "David"
                ]
            },
            "number_of_occurrences": 2,
            "grouping_columns": [
                "id"
            ],
            "column_differences": [
                "name"
            ],
            "consise_information": {
                "id": [
                    3,
                    3
                ],
                "name": [
                    "Charlie",
                    "David"
                ],
                "source": [
                    "df_0",
                    "df_1"
                ]
            },
            "row_with_source": {
                "age": [
                    35,
                    35
                ],
                "id": [
                    3,
                    3
                ],
                "name": [
                    "Charlie",
                    "David"
                ],
                "source": [
                    "df_0",
                    "df_1"
                ]
            }
        }
    ]
}
```
As You can see the `DataReport` will have some basic information like:
- `df0_length` and `df1_length` which are the lengths of the dataframes
- `df0_name` and `df1_name` which are the names of the dataframes
- `comparable_columns` which are the columns that are comparable between the two dataframes
- `colum_differences` which are the columns that have differences between the two dataframes and the type of difference
- `row_differences` which are the rows that have differences between the two dataframes and the type of difference

If you look closely at `column_differences` you will see that we always refrence the first dataframe as the source. Also You can see that there are different type of differences with more detailed information about the differences.

If you look at `row_differences` you will see that there are also multiple type of differences. We generally have two types of differences:
- `RowDifference` which is a difference between two rows that couldn't be grouped or there wan't any grouping
- `RowGroupDifference` which is a difference between two groups of rows that were grouped by the `grouping_columns`

When talking about `RowDifference` we have the following information:
- `source` which is the source of the row
- `row` which is original row that is different (keep on mind that it contains only the comparable columns, look at parameter `comparable_columns`)
- `number_of_occurrences` which is the number of times this row is present in the source
- `difference_type` which is the type of difference (`MISSING_ROW` says that the row is missing in the other dataframe)
- `more_information` which is more information about the difference (usually `None`)

When talking about `RowGroupDifference` we have the following information:
- `sources` which are the sources of the grouped rows (grouped by `grouping_columns`)
- `row` which are rows present in that group (keep on mind that it contains only the comparable columns, look at parameter `comparable_columns`)
- `number_of_occurrences` which is the number of times this difference is present in all sources (**total**)
- `grouping_columns` which are the columns used to group the rows
- `column_differences` which are the columns that are different
- `consise_information` which is a dictionary with more information about the differences containing the grouping columns, source of the row and the `column_differences` from original data (parameter `row`)
- `row_with_source` which is basically the same as the `row` but with the source

Now when we have differences we can get tabular information about those differences:
```python
print(get_dataframe(report))
```
Output:
```python
shape: (3, 4)
┌─────┬─────┬─────────┬────────┐
│ age ┆ id  ┆ name    ┆ source │
│ --- ┆ --- ┆ ---     ┆ ---    │
│ i64 ┆ i64 ┆ str     ┆ str    │
╞═════╪═════╪═════════╪════════╡
│ 26  ┆ 4   ┆ George  ┆ df_0   │
│ 35  ┆ 3   ┆ Charlie ┆ df_0   │
│ 35  ┆ 3   ┆ David   ┆ df_1   │
└─────┴─────┴─────────┴────────┘
```

We also have an option to gather more information about the differences:

+ Get the number of row differences:
```python
from data_fingerprint.src.utils import get_number_of_row_differences

print(get_number_of_row_differences(report))
```
Output:
```
3
```

+ Get the number of differences per source:
```python
from data_fingerprint.src.utils import get_number_of_differences_per_source

print(get_number_of_differences_per_source(report))
```
Output:
```
{'df_0': 2, 'df_1': 1}
```

+ Get the ratio of differences per source:
```python
from data_fingerprint.src.utils import get_ratio_of_differences_per_source

print(get_ratio_of_differences_per_source(report))
```
Output:
```
{'df_0': 0.6666666666666666, 'df_1': 0.3333333333333333}
```

+ Get column difference ratio:
```python
from data_fingerprint.src.utils import get_column_difference_ratio

print(get_column_difference_ratio(report))
```
Output:
```
{'age': 0.25, 'name': 0.5, 'id': 0.25}
```

## License

This project is licensed under the GPLv3 License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## Contact

For any questions or feedback, please contact me over github.