---
title: 'Machine Learning Notes (5): Overfitting and Underfitting'
description: 'Recording the machine learning progress'
author: LYGreen
date: 2026-02-23T19:50:57+08:00
updated: 2026-02-23T19:50:57+08:00
category: Artificial Intelligence
tags: ['AI', 'Python', 'Machine Learning']
---

## Overfitting and Underfitting

### Underfitting

**Underfitting**: The model performs poorly even on the training dataset and fails to capture the underlying patterns of the data.

Causes:
- The model is too simple.
- There are too few features.
- Insufficient training time.

In decision trees, this may be caused by a tree depth that is too shallow.

### Overfitting

**Overfitting**: The model performs perfectly on the training dataset but performs poorly when encountering new data.

Causes:
- The model is too complex.
- The dataset is too small, causing the model to memorize noise.
- The learning rate is too high, or there are too many training iterations.

In decision trees, this may be caused by a tree depth that is too deep.

## Example (Kaggle Course)

```python
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
```

Output:
```
Max leaf nodes: 5                Mean Absolute Error:  347380
Max leaf nodes: 50               Mean Absolute Error:  258171
Max leaf nodes: 500              Mean Absolute Error:  243495
Max leaf nodes: 5000             Mean Absolute Error:  255575
```
As shown here, the Mean Absolute Error (MAE) is at its minimum when the number of leaf nodes is 500.

## Resources
ChatGPT
Gemini
[Kaggle Course](https://www.kaggle.com/code/dansbecker/underfitting-and-overfitting)