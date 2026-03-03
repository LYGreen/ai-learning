import pandas as pandas
df = pandas.read_csv('people.csv')
print("First few row:")
print(df.head())
print("\nDataFrame info:")
print(df.info())