---

title: 'Pandas Notes (2): Indexing, Selection, and Assignment'

description: 'Methods for using the Python data processing library'

author: LYGreen

date: 2026-03-02T16:29:08+08:00

updated: 2026-03-02T16:29:08+08:00

category: Data Analysis

tags: \['Pandas', 'Python']

---



\## Basic Indexing

\- Select a single column (returns a Series): ```df\['col']```

\- Select multiple columns (returns a DataFrame): ```df\[\['col1', 'col2']]```

\- Row slicing: ```df\[0:3]```



> Row slicing does not support single values; for example, ```df\[0]``` will result in an error.



\## Explicit Accessors



\### Label-based Selection

Syntax: ```df.loc\[row\_label, column\_label]``` or ```df.loc\[row, column\_label]```

Examples:

```python

\# Select the 'column' value from the row labeled 1

df.loc\[1, 'column']



\# Select the 'column2' column for all rows

df.loc\[:, 'column2']

```



> When using row labels for slicing, the range is inclusive of the end point. For example, ```df.loc\[0:2, 'column']``` selects rows 0, 1, and 2.



\### Position-based Selection

Syntax: ```df.iloc\[row\_pos, column\_pos]```

Examples:

```python

\# Select the last row

df.iloc\[-1, :]



\# Select the first 5 rows and the first 3 columns

df.iloc\[0:5, 0:3]

```



> When using position-based selection, the range follows standard Python slicing (inclusive of the start, exclusive of the end).



\### Boolean Indexing

\- Single condition: ```df\[df\['column'] > 5]```

\- Multiple conditions: You must use bitwise operators \& (and), | (or), ~ (not), and each condition must be wrapped in parentheses.

Example:

```python

\# Records where total\_bill > 20 and the day is Saturday

mask = (df\['total\_bill'] > 20) \& (df\['day'] == 'Sat')

print(df\[mask])

```



\### Advanced Functional Indexing



\- ```.isin()```: Matches multiple values from a list.

```python

\# Select data for Saturday and Sunday

df\[df\['day'].isin(\['Sat', 'Sun'])]

```



\- ```.query()```: Uses string expressions (similar to SQL).

```python

\# The previous example converted to a string expression

df.query("total\_bill > 20 and day == 'Sat'")

```



\- ```.at``` and ```.iat```: Specifically used for getting or modifying a single cell.

```python

df.at\[0, 'total\_bill'] = 21

```



\## Assignment



\### Full Assignment

```python

\# Create a new 'test' column and assign True to all rows

df\['test'] = True



\# Assign the string 'test' to all rows in the 'column' column

df\['column'] = 'test'

```



\### Conditional Assignment

```python

\# Assign level as 'VIP' when the price exceeds 40

df.loc\[df\['total\_bill'] > 40, 'level'] = 'VIP'



\# Assign others as 'Normal'

df.loc\[df\['total\_bill'] <= 40, 'level'] = 'Normal'

```



\### Logical Assignment

Using ```np.where```:

```python

df\['sex\_short'] = np.where(df\['sex'] == 'Male', 'M', 'F')



df.loc\[df\['smoker'] == 'Yes', 'total\_bill'] += 5

```



\### Dictionary Mapping Assignment

\- Using ```map```:

```python

\# Scenario: Translate days of the week to Chinese

day\_map = {'Sun': '周日', 'Sat': '周六', 'Fri': '周五', 'Thur': '周四'}

df\['day\_cn'] = df\['day'].map(day\_map)

```



\- Using ```replace```:

```python

\# Scenario: Replace specific values directly in the original column

df\['smoker'] = df\['smoker'].replace({'Yes': 1, 'No': 0})

```



\### Danger: Chained Assignment

```python

\# WRONG (may not take effect or may throw a warning/error):

df\[df\['total\_bill'] > 40]\['tip'] = 0 



\# CORRECT (Explicit positioning):

df.loc\[df\['total\_bill'] > 40, 'tip'] = 0

```



\## Dataset Example

Example dataset: \[tip.csv](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv)

