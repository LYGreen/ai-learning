---
title: 'Machine Learning Notes (4): Model Validation'
description: 'Documentation of the machine learning learning process'
author: LYGreen
date: 2026-02-17T19:30:39+08:00
updated: 2026-02-17T19:30:39+08:00
category: Artificial Intelligence
tags: ['AI', 'Python', 'Machine Learning']
---

## Model Validation

**Model Validation** is a core technique used to ensure a model's generalization ability, accuracy, and stability by evaluating its performance on an independent dataset, thereby preventing overfitting.

### Mean Absolute Error

The mathematical formula for Mean Absolute Error (MAE) is: $$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

#### Validating Seen Data

In this section, we use **Mean Absolute Error (MAE)** for validation.

This content is a follow-up to the previous lesson. The Kaggle environment has already provided the code for loading the dataset and training; our task is to validate the data.

```python
# Data Loading Code Hidden Here
import pandas as pd

# Load data
melbourne_file_path = './input/melb_data.csv' # Path modified locally
melbourne_data = pd.read_csv(melbourne_file_path) 
# Filter rows with missing price values
filtered_melbourne_data = melbourne_data.dropna(axis=0)
# Choose target and features
y = filtered_melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']
X = filtered_melbourne_data[melbourne_features]

from sklearn.tree import DecisionTreeRegressor
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(X, y)
```

Next, we use Mean Absolute Error to validate the values predicted by the model:
```python
from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X)
print(mean_absolute_error(y, predicted_home_prices))
```
Output:
```
434.71594577146544
```

#### Validating Unseen Data

```python
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

# Split the dataset into training and validation data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

melbourne_model = DecisionTreeRegressor()

melbourne_model.fit(train_X, train_y)

val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))
```
Output:
```
254442.03163331182
```

As observed, the model is more accurate when predicting data it has already "seen," but performance becomes significantly less accurate when predicting data it has not encountered before.

## Resources
[Kaggle Course](https://www.kaggle.com/code/dansbecker/model-validation)