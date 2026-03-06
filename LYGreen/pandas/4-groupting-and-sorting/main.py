import pandas as pd

df = pd.DataFrame({
    "name": ["Alice","Alice","Bob","Bob","Cindy","Cindy"],
    "class": ["1","1","1","1","2","2"],
    "subject": ["Math","English","Math","English","Math","English"],
    "score": [90,85,88,80,92,87]
})

print(df.groupby("class")["score"].mean())
print(df.groupby("class").count())
print(df.groupby("class").size())
print(df.groupby(["name", "class"])["score"].sum())
print(df.groupby(["class", "subject"])["score"].agg(["mean", "max", "min"]))
print(df.groupby(["class", "subject"])["score"].agg(lambda x: x.max() - x.min()))

print(df.sort_values("score", ascending=False))
print(df.sort_values(["class", "score"]))
print(df.sort_values("score", ascending=False).groupby("class").head(1))
