---
title: 'Machine Learning Notes (3): Your First Model'
description: 'Documentation of the machine learning learning process'
author: LYGreen
date: 2026-02-17T18:17:22+08:00
updated: 2026-02-17T18:17:22+08:00
category: Artificial Intelligence
tags: ['AI', 'Python', 'Machine Learning']
---

## Your First Model

### Environment
Install the necessary dependencies:
```bash
pip install pandas scikit-learn
```

### Prepare the Dataset
There is an [official Kaggle dataset](https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot); download it and place it in the `./input/` directory.

Code:
```python
import pandas as pd

melbourne_file_path = './input/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

print(melbourne_data.columns)
```

Output:
```
Index(['Suburb', 'Address', 'Rooms', 'Type', 'Price', 'Method', 'SellerG',
       'Date', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car',
       'Landsize', 'BuildingArea', 'YearBuilt', 'CouncilArea', 'Lattitude',
       'Longtitude', 'Regionname', 'Propertycount'],
      dtype='str')
```

This dataset contains many rows with missing values, which can be seen by printing:
```python
print(melbourne_data.isna().any(axis=1))
```

Output:
```
0         True
1        False
2        False
3         True
4        False
         ...
13575     True
13576     True
13577     True
13578     True
13579     True
Length: 13580, dtype: bool
```

For now, we will not handle the missing entries; we will simply drop the rows that contain them:
```python
melbourne_data = melbourne_data.dropna(axis=0)
```

### Selecting Features and Prediction Target

Select the columns to be input into the model and the column you want the model to predict. The purpose of these two is to allow the model to adjust its internal parameters so that it can output correct prediction values.

```python
X = melbourne_data[['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']]
y = melbourne_data.Price
```

### Building the Model

We will use a Decision Tree to build a model here.


```python
from sklearn.tree import DecisionTreeRegressor

# Create a Decision Tree model
melbourne_model = DecisionTreeRegressor(random_state=1)

# Train the model
melbourne_model.fit(X, y)
```

### Model Prediction
```python
print("Input values for the model:")
print(X.head())
print("Prediction results:")
print(melbourne_model.predict(X.head()))
```

Output:
```
Input values for the model:
   Rooms  Bathroom  Landsize  Lattitude  Longtitude
1      2       1.0     156.0   -37.8079    144.9934
2      3       2.0     134.0   -37.8093    144.9944
4      4       1.0     120.0   -37.8072    144.9941
6      3       2.0     245.0   -37.8024    144.9993
7      2       1.0     256.0   -37.8060    144.9954
Prediction results:
[1035000. 1465000. 1600000. 1876000. 1636000.]
```

> In practice, the dataset should be split into two parts: one for training and one for testing. However, here we directly used the training data for prediction. If predicting data that the model has not seen before, the accuracy will typically decrease.

## Resources
[Kaggle Course](https://www.kaggle.com/code/dansbecker/your-first-machine-learning-model)