import pandas as pandas

data = {
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['NY', 'LA', 'Chicago']
}

df = pandas.DataFrame(data)
print(df)
df.to_csv('people.csv', index=False)
print("\nSaved to people.csv")