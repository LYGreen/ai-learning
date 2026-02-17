import pandas as pd

melbourne_file_path = './input/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

melbourne_data = melbourne_data.dropna(axis=0)

X = melbourne_data[['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']]
y = melbourne_data.Price

from sklearn.tree import DecisionTreeRegressor

# 创建一个决策树模型
melbourne_model = DecisionTreeRegressor(random_state=1)

# 训练模型
melbourne_model.fit(X, y)

print("输入给模型的值")
print(X.head())
print("预测结果")
print(melbourne_model.predict(X.head()))
