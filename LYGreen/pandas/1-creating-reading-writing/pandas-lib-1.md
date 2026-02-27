---
title: 'Pandas Notes (1): Creating, Reading, and Writing'
description: 'Methods for using the Python data processing library'
author: LYGreen
date: 2026-02-13T16:30:46+08:00
updated: 2026-02-24T14:02:23+08:00
category: Data Analysis
tags: ['Pandas', 'Python']
---

## Creating Data

### Creating Data with DataFrame

#### Via Dictionary
```python
import pandas as pd

df = pd.DataFrame({
    "name": ["A", "B", "C"],
    "age": [20, 21, 22]
})

print(df)
```
**Output:**
```
  name  age
0    A   20
1    B   21
2    C   22
```

#### From a List
```python
df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(df)
```
**Output:**
```
   0  1  2              
0  1  2  3
1  4  5  6
2  7  8  9
```

### Creating Data with Series
```python
df = pd.Series([1, 2, 3], index=['A', 'B', 'C'], name='numbers')
print(df)
```
**Output:**
```
A    1
B    2
C    3
Name: numbers, dtype: int64
```

### Creating Data with NumPy
```python
import pandas as pd
import numpy as np

arr = np.random.randn(3, 3)
df = pd.DataFrame(arr, columns=["A", "B", "C"])
print(df)
```
**Output:**
```
          A         B         C
0 -0.132643 -0.430935 -0.051926
1 -1.337611 -1.457143 -0.088758
2  0.135328  1.975916  0.105259
```

## Parameters

### DataFrame
When creating data, you can specify row labels (index) and column names:
```python
df = pd.DataFrame([
    [1, 2, 3], 
    [4, 5, 6]
], columns=['a', 'b', 'c'], index=['first', 'second'])
print(df)
```
**Output:**
```
        a  b  c
first   1  2  3
second  4  5  6
```

### Series
Since a Series represents a single column, it uses the `name` parameter instead of `columns`:
```python
df = pd.Series([1, 2, 3, 4, 5], name='A', index=['a', 'b', 'c', 'd', 'e'])
print(df)
```
**Output:**
```
a    1
b    2
c    3
d    4
e    5
Name: A, dtype: int64
```

## Reading Data

### Read CSV
```python
df = pd.read_csv("data.csv")
```

### Read Excel
```python
df = pd.read_excel("data.xlsx")
```

### Read SQL (via SQLite)
```python
import sqlite3
import pandas as pd

conn = sqlite3.connect('database.db')
df = pd.read_sql("SELECT * FROM table_name", conn)
```

## Writing Data
When writing data, if `index=False` is not specified, the index will be included in the output.

### Write to CSV
```python
df.to_csv("output.csv", index=False)
```

### Write to Excel
```python
df.to_excel("output.xlsx", index=False)
```

### Write to JSON
```python
df.to_json("output.json")
```

### Write to SQL
```python
df.to_sql("table_name", conn, if_exists="replace")
```

## Resources
* ChatGPT
* [Kaggle Course](https://www.kaggle.com/code/residentmario/creating-reading-and-writing)