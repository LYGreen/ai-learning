from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd

melbourne_file_path = './input/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 

filtered_melbourne_data = melbourne_data.dropna(axis=0)

y = filtered_melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']
X = filtered_melbourne_data[melbourne_features]

# 从数据集中分离训练数据和验证数据
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

melbourne_model = DecisionTreeRegressor()

melbourne_model.fit(train_X, train_y)

val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))
