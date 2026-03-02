import pandas as pd
import numpy as np

tips = pd.read_csv("./input/tips.csv")

result = tips.columns

# 显示列名
print("列名：" + str(result.to_list()))

# 选择一列
result = tips['total_bill']
# print(result)

# 选择多列
result = tips[['total_bill', 'tip']]
# print(result)

# 行切片
result = tips[0:3]
# print(result)

# loc 选取
result = tips.loc[1, 'total_bill']
# print(result)

# iloc 选取
result = tips.iloc[:, 0:4]
# print(result)

# 布尔选取
result = tips[tips['smoker'] == 'Yes']
# print(result)

# .isin()
result = tips[tips['day'].isin(['Sat', 'Sun'])]
# print(result)

# .query()
result = tips.query("day == 'Sat' or day == 'Sun'")
# print(result)

# 赋值
tips['day'] = 'Sun'
# print(result)

# 条件赋值
tips.loc[tips['total_bill'] > 40, 'level'] = 'VIP'
# print(tips)
tips.loc[tips['smoker'] == 'Yes', 'total_bill'] += 5
# print(tips)
