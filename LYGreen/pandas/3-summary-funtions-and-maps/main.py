import pandas as pd

# 测试数据集
ds = pd.DataFrame({
    "a": [5, 3, 6],
    "b": [9, 5, 6],
    "c": [1, 8, 0],
    "d": [2, 6, 1],
    "e": [7, 1, 3],
})

print(ds.describe())

s = pd.Series([1, 2, 3])
s = s.map(lambda x: x * 10)
print(s)

s = pd.Series(['a', 'b', 'c', 'a', 'c', 'c', 'b'])
s = s.map({
    'a': 1,
    'b': 2,
    'c': 3
})
print(s)

s = pd.DataFrame({
    "a": [5, 3, 6],
    "b": [9, 5, 6],
    "c": [1, 8, 0],
    "d": [2, 6, 1],
    "e": [7, 1, 3],
})
s = s.applymap(lambda x: x * 10)
print(s)
