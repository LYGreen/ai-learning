import pandas as pd
import numpy as np

df = pd.DataFrame({
    "name": ["Alice", "Bob"],
    "age": [20, 21],
    "score": [90.5, 88.0],
    "passed": [True, False]
})

# print(df)

# print(df.info())

# print(df["name"].dtypes)

# print(df["score"].astype("int"))

df = pd.DataFrame({
    "name": ["Alice", "Bob", None],
    "score": [90, np.nan, 85]
})

# print(df)

# print(df.isna())

# print(df.isnull())

# print(df.isna().sum())

# print(df.isnull())

# print(df[df.isnull()])

# print(df[pd.isnull(df['name'])])

# print(df.dropna())

# print(df.fillna(0))

print(df.replace(np.nan, 'Unknown'))
