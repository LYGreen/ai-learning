---
title: 'Pandas Notes (4): Grouping and Sorting'
description: 'Methods for using the Python data processing library'
author: LYGreen
date: 2026-03-06T07:57:28+08:00
updated: 2026-03-06T07:57:28+08:00
category: Data Analysis
tags: ['Pandas', 'Python']
---

## Pandas Grouping
Grouping is used to **split data into groups based on the values of a specific column, and then perform statistics on each group**. The core philosophy of grouping is **Split**, **Apply**, and **Combine**.

### Basic Usage
- Single column grouping: ```df.groupby('column').func()``` or ```df.groupby('column')['apply_column'].func()```
- Multi-column grouping: ```df.groupby(['col1', 'col2']).func()``` or ```df.groupby(['col1', 'col2'])['apply_column'].func()```

### Example Code

#### Data
```python
import pandas as pd

df = pd.DataFrame({
    "name": ["Alice","Alice","Bob","Bob","Cindy","Cindy"],
    "class": ["1","1","1","1","2","2"],
    "subject": ["Math","English","Math","English","Math","English"],
    "score": [90,85,88,80,92,87]
})
```
Output:
```
    name class  subject  score
0  Alice     1     Math     90
1  Alice     1  English     85
2    Bob     1     Math     88
3    Bob     1  English     80
4  Cindy     2     Math     92
5  Cindy     2  English     87
```
#### Calculate average value per group
```python
df.groupby("class")["score"].mean()
```
Output:
```
class
1    85.75
2    89.50
Name: score, dtype: float64
```

#### Calculate quantity per group
```python
df.groupby("class").count()

df.groupby("class").size()
```
Output:
```
       name  subject  score
class
1         4        4      4
2         2        2      2

class
1    4
2    2
dtype: int64
```

#### Multi-column grouping
```python
df.groupby(["name", "class"])["score"].sum()
```
Output:
```
name   class
Alice  1        175
Bob    1        168
Cindy  2        179
Name: score, dtype: float64
```

#### Multiple statistics
```python
df.groupby(["class", "subject"])["score"].agg(["mean", "max", "min"])
```
Output:
```
               mean  max  min
class subject
1     English  82.5   85   80
      Math     89.0   90   88
2     English  87.0   87   87
      Math     92.0   92   92
```

#### Custom statistical function
```python
df.groupby(["class", "subject"])["score"].agg(lambda x: x.max() - x.min())
```
Output:
```
class  subject
1      English    5
       Math       2
2      English    0
       Math       0
Name: score, dtype: float64
```

## Pandas Sorting

### Basic Usage
Pandas primarily provides two sorting methods: sorting by **index** and sorting by **values**.

- Sort by values: ```.sort_values("column")```
- Sort by index: ```.sort_index()```

> Parameters: ```ascending=False``` stands for descending, ```ascending=True``` stands for ascending (default)

### Example Code

#### Single column sort
```python
df.sort_values("score")
```
Output:
```
    name class  subject  score
3    Bob     1  English     80
1  Alice     1  English     85
5  Cindy     2  English     87
2    Bob     1     Math     88
0  Alice     1     Math     90
4  Cindy     2     Math     92
```

#### Descending sort
```python
df.sort_values("score", ascending=False)
```
Output:
```
    name class  subject  score
4  Cindy     2     Math     92
0  Alice     1     Math     90
2    Bob     1     Math     88
5  Cindy     2  English     87
1  Alice     1  English     85
3    Bob     1  English     80
```

#### Multi-column sort
```python
df.sort_values(["class", "score"])
```
Output:
```
    name class  subject  score
3    Bob     1  English     80
1  Alice     1  English     85
2    Bob     1     Math     88
0  Alice     1     Math     90
5  Cindy     2  English     87
4  Cindy     2     Math     92
```

## Combining Grouping and Sorting

```python
df.sort_values("score", ascending=False).groupby("class").head(1)
```
Output:
```
    name class subject  score
4  Cindy     2    Math     92
0  Alice     1    Math     90
2    Bob     1    Math     88
```
> Here, sorting is performed first, followed by grouping; you can use ```.head()``` to see the steps, for example:
> 1. Sort first
> ```df.sort_values("score", ascending=False).head()```
> Output:
> ```
>     name class  subject  score
> 4  Cindy     2     Math     92
> 0  Alice     1     Math     90
> 2    Bob     1     Math     88
> 5  Cindy     2  English     87
> 1  Alice     1  English     85
> ```
> 
> 2. Then group
> ```df.sort_values("score", ascending=False).groupby("name").head()```
> Output:
> ```
>     name class  subject  score
> 4  Cindy     2     Math     92
> 0  Alice     1     Math     90
> 2    Bob     1     Math     88
> 5  Cindy     2  English     87
> 1  Alice     1  English     85
> 3    Bob     1  English     80
> ```
> 
> 3. Only output the first of each group
> ```df.sort_values("score", ascending=False).groupby("name").head(1)```
> Output:
> ```
>     name class subject  score
> 4  Cindy     2    Math     92
> 0  Alice     1    Math     90
> 2    Bob     1    Math     88
> ```

## Resources
- Gemini
- ChatGPT
- [Kaggle Course](https://www.kaggle.com/code/residentmario/grouping-and-sorting)