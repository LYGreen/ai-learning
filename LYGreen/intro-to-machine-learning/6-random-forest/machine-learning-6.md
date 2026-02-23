---
title: 'Machine Learning Notes (6): Random Forest'
description: 'Recording the machine learning progress'
author: LYGreen
date: 2026-02-23T21:31:32+08:00
updated: 2026-02-23T21:31:32+08:00
category: Artificial Intelligence
tags: ['AI', 'Python', 'Machine Learning']
---

## Random Forest

**Random Forest** is an **Ensemble Learning** algorithm. By building many uncorrelated decision trees and aggregating their results, it achieves a more accurate and robust model.

### How It Works

**Random Sampling of Data**: Samples are randomly drawn from the original training set with replacement (Bootstrap). Each tree sees slightly different data; some trees may focus on specific samples while others may not see them at all.
- Sampling with replacement.
- Each tree uses a different subset of data.

**Random Feature Selection**: When a decision tree splits a node, instead of choosing the best from all features, it first randomly picks a few features and then selects the best among those. This prevents certain dominant features from overshadowing all decision trees.
- Does not use all features.
- Randomly selects only a portion of features.

> Note: Features here refer to the columns of the dataset.

**Voting or Averaging**:
- Classification problems -> Majority voting.
- Regression problems -> Average of results.

Compared to a single decision tree, Random Forest effectively solves the problem of overfitting.


## Pros and Cons

**Pros**:
- Not prone to overfitting.
- Suitable for high-dimensional data.

**Cons**:
- Slow training speed.
- Large model size.
- Higher computational overhead and memory consumption.

## Mathematical Essence
The essence of Random Forest is performing:
$$ \text{Final Prediction} = \frac{1}{T} \sum_{t=1}^{T} h_t(x) $$
Where:
- $ T $ = Number of trees.
- $ h_t(x) $ = Prediction of the $ t $-th tree.

## Example (Kaggle Course Code)
Loading the dataset:
```python
import pandas as pd
    
# Load data
melbourne_file_path = './input/melb_data.csv' # Path updated
melbourne_data = pd.read_csv(melbourne_file_path) 
# Filter out rows with missing values
melbourne_data = melbourne_data.dropna(axis=0)
# Select target and features
y = melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]

from sklearn.model_selection import train_test_split

# Split the dataset into training and validation sets using train_test_split
# random_state ensures the split is reproducible
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
```
Training:
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))
```
Output:
```
191669.7536453626
```
Comparing this to the value ```254442.03163331182``` from *Machine Learning Notes (4): Model Validation*, the Mean Absolute Error (MAE) has significantly decreased.

## Resources
ChatGPT
Gemini
[Kaggle Course](https://www.kaggle.com/code/dansbecker/random-forests)