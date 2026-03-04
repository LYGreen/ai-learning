---
title: 'Pandas Notes (3): Statistical Functions and Mapping Operations'
description: 'Usage of Python data processing library'
author: LYGreen
date: 2026-03-03T16:20:46+08:00
updated: 2026-03-03T16:20:46+08:00
category: Data Analysis
tags: ['Pandas', 'Python']
---

## Statistical Functions

Common statistical functions include:
| Function | Description |
|---|---|
| ```.sum()``` | Sum |
| ```.mean()``` | Mean (Average) |
| ```.median()``` | Median |
| ```.max()``` | Maximum |
| ```.min()``` | Minimum |
| ```.std()``` | Standard Deviation |
| ```.count()``` | Count (Non-null values) |
| ```.unique()``` | Unique values |
| ```.value_counts()``` | Frequency counts |

**Example:**
```python
import pandas as pd

ds = pd.DataFrame({
    "a": [5, 3, 6],
    "b": [9, 5, 6],
    "c": [1, 8, 0],
    "d": [2, 6, 1],
    "e": [7, 1, 3],
})

print(ds.mean())
```
**Output:**
```
a    4.666667
b    6.666667
c    3.000000
d    3.000000
e    3.666667
dtype: float64
```

> Statistical functions can also be applied to a specific column, for example: ```ds.a.sum()```. Output：```14```

## Description
Use ```.describe()``` to view common statistical information, for example:
```python
ds = pd.DataFrame({
    "a": [5, 3, 6],
    "b": [9, 5, 6],
    "c": [1, 8, 0],
    "d": [2, 6, 1],
    "e": [7, 1, 3],
})

print(ds.describe())
```
**Output:**
```
              a         b         c         d         e
count  3.000000  3.000000  3.000000  3.000000  3.000000
mean   4.666667  6.666667  3.000000  3.000000  3.666667
std    1.527525  2.081666  4.358899  2.645751  3.055050
min    3.000000  5.000000  0.000000  1.000000  1.000000
25%    4.000000  5.500000  0.500000  1.500000  2.000000
50%    5.000000  6.000000  1.000000  2.000000  3.000000
75%    5.500000  7.500000  4.500000  4.000000  5.000000
max    6.000000  9.000000  8.000000  6.000000  7.000000
```
> The ```25%, 50%, 75%``` values here represent **Quantiles**.

## Mapping Operations

### map()
The ```.map()``` method modifies cells and can only be used on a **Series**. Here is a code example:
```python
s = pd.Series([1, 2, 3])
s = s.map(lambda x: x * 10)
print(s)
```
**Output:**
```
0    10
1    20
2    30
dtype: int64
```
It can also be used for dictionary mapping:
```python
s = pd.Series(['a', 'b', 'c', 'a', 'c', 'c', 'b'])
s = s.map({
    'a': 1,
    'b': 2,
    'c': 3
})
print(s)
```
**Output:**
```
0    1
1    2
2    3
3    1
4    3
5    3
6    2
dtype: int64
```

### apply()
The ```apply()``` method can be used for both **Series** and **DataFrame**. It can act on an entire row, an entire column, or every individual element. For example:
```python
s = pd.DataFrame({
    "a": [5, 3, 6],
    "b": [9, 5, 6],
    "c": [1, 8, 0],
    "d": [2, 6, 1],
    "e": [7, 1, 3],
})
s = s.apply(lambda x: x.max())
print(s)
```
**Output:**
```
a    6
b    9
c    8
d    6
e    7
dtype: int64
```
