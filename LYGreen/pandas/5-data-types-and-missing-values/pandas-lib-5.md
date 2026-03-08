---
title: 'Pandas Notes (5): Data Types and Missing Values'
description: 'Usage of Python data processing library'
author: LYGreen
date: 2026-03-08T16:55:27+08:00
updated: 2026-03-08T16:55:27+08:00
category: Data Analysis
tags: ['Pandas', 'Python']
---

## Data Types

### Common Data Types

| Type | Meaning | Example |
|---|---|---|
| `int64` | Integer | 1, 2, 3 |
| `float64` | Floating-point number | 3.14 |
| `object` | String or mixed types | "Alice" |
| `bool` | Boolean | True / False |
| `datetime64[ns]` | Date and Time | 2026-01-01 |
| `category` | Categorical variable | Yes / No |

### Data Example
```python
import numpy as np

df = pd.DataFrame
```
Output:
```
    name  age  score  passed
0  Alice   20   90.5    True
1    Bob   21   88.0   False
```

### Checking All Column Types
Use `.dtypes` to view the types of all columns:
```python
df.dtypes
```
Output:
```
name          str
age         int64
score     float64
passed       bool
dtype: object
```

### Checking a Specific Column Type
Use `.column.dtypes` or `['column'].dtypes` to check a specific column's type:
```python
df.name.dtypes
# OR
df["name"].dtypes
```
Output:
```
str
```

### Viewing Detailed Information
Use `.info()` to view detailed information:
```python
df.info()
```
Output:
```
<class 'pandas.DataFrame'>
RangeIndex: 2 entries, 0 to 1
Data columns (total 4 columns):
 #   Column  Non-Null Count  Dtype
---  ------  --------------  -----
 0   name    2 non-null      str
 1   age     2 non-null      int64
 2   score   2 non-null      float64
 3   passed  2 non-null      bool
dtypes: bool(1), float64(1), int64(1), str(1)
memory usage: 182.0 bytes
None
```

### Type Conversion
Use `.astype()` to perform type conversion:
```python
df.score.astype("int")
# OR
df["score"].astype("int")
```
Output:
```python
0    90 
1    88
Name: score, dtype: int64
```

## Missing Values

### Representation
| Representation | Meaning |
|---|---|
| `NaN` | Numerical missing value |
| `None` | Python null value |
| `NaT` | Time type missing value |

### Example Data
```python
import numpy as np

df = pd.DataFrame({
    "name": ["Alice", "Bob", None],
    "score": [90, np.nan, 85]
})

print(df)
```
Output:
```
    name  score
0  Alice   90.0
1    Bob    NaN
2    NaN   85.0
```

### Detecting Missing Values
Use `.isna()` or `.isnull()` to determine if data is missing:
```python
df.isna()
# OR
df.isnull()
```
Output:
```
    name  score
0  False  False
1  False   True
2   True  False
```
> `.isna()` and `.isnull()` actually return a table (DataFrame) of the same size as the original data, setting items with data to False and missing items to True.

### Counting Missing Values per Column
```python
df.isna().sum()
```
Output:
```
name     1
score    1
dtype: int64
```

### Missing Values and Indexing

- **Example 1**: `df[df.isnull()]`  
`df.isnull()` returns a table of True/False values. When placed into the `df` index, it retains original values for True items and changes False items to `NaN`. Since the original values for True items were already `NaN`, the result is a table entirely filled with `NaN`.
> Compare this with `df[df.notnull()]`.
```python
df[df.isnull()]
```
Output:
```
  name  score
0  NaN    NaN
1  NaN    NaN
2  NaN    NaN
```

- **Example 2**: `df[pd.isnull(df['name'])]`  
`pd.isnull(df['name'])` returns a single column (Series) of True/False values. When placed in the `df` index, it retains rows where the value is True and removes rows where it is False.

```python
df[pd.isnull(df['name'])]
```
Output:
```
  name  score
2  NaN   85.0
```

### Handling Missing Values
- **Dropping Missing Rows** Use `.dropna()` to delete rows containing missing values:
```python
df.dropna()
```
Output:
```
    name  score
0  Alice   90.0
```

- **Filling Missing Rows** Use `.fillna()` to fill missing rows:
```python
df.fillna(0) # Example fill with 0
```
Output:
```
    name  score
0  Alice   90.0
1    Bob    0.0
2      0   85.0
```

- **Replacing Values** Use `.replace()` to substitute values:
```python
df.replace(np.nan, 'Unknown')
```
Output:
```
      name    score
0    Alice     90.0
1      Bob  Unknown
2  Unknown     85.0
```

## Resources
- Gemini
- ChatGPT
- [Kaggle Course: Pandas - Data Types and Missing Values](https://www.kaggle.com/code/residentmario/data-types-and-missing-values#Introduction)