---
title: 'Machine Learning Notes (1): How Models Work'
description: 'Recording the learning process of machine learning'
author: LYGreen
date: 2026-02-13T20:27:55+08:00
updated: 2026-02-18T01:24:56+08:00
category: Artificial Intelligence
tags: ['AI', 'Python', 'Machine Learning']
---

## Machine Learning
**Machine Learning** is about letting computers learn patterns from data, rather than executing tasks through hard-coded instructions.

## Models
A model is composed of algorithms and data, similar to a function $y = f(x)$. By feeding known conditions into the model, the model provides predicted results.

## Core Workflow of a Model
### Input Data
Before training, we feed the model a large amount of **training data**. The model tries to find the mathematical relationship between **inputs** and **labels**. For example:

| Input Data | Label |
|---|---|
| House Size | House Price |
| Historical Weather | Tomorrow's Weather |
| User Browsing History | Ad Click-through |
| Email Content | Spam or Not |

> Training data: Data sent to the model containing both **inputs** and **labels (correct output results)**.

### Inference
The model performs calculations during this stage, for example:
$$y = wx+b$$
- $x$: Input
- $w, b$: Model parameters (adjusted by the model)
- $y$: Prediction

By using the provided **inputs** and **labels** during the learning phase, the model adjusts its $w$ and $b$ values. Once the model finds the "pattern," it can handle unseen data.

### Training
1. Prepared data:
    | Study Time $x$ (hours) | Score $y$ (points) |
    |---|---|
    | 1 | 50 |
    | 2 | 60 |
    | 3 | 70 |
    | 4 | 80 |

2. Initially, model parameters are random, so the output is also random. The model calculates the gap between the output and the label. For example:
    $$y = 10x + 20$$
    When $x = 3$, the correct value should be $70$, but the untrained model predicts $50$.
    $$error = \text{Actual Score} - \text{Model Prediction} = 20$$
    The process of calculating the loss is called the **Loss Function**.

3. The model adjusts its parameters ($w$ and $b$ values) based on the loss function.
    $$y = 11.5x + 22.4$$
    This adjustment process is called **Gradient Descent**.

4. By repeating steps 1–3 multiple times, the model reaches a final adjusted result:
    $$y = 10x + 40$$

## Three Main Types of Machine Learning
Models are generally classified into three types based on their learning style:

| Type | Characteristics | Scenarios |
|---|---|---|
| **Supervised Learning** | Has "standard answers"; the model constantly moves toward them. | Spam identification, house price prediction |
| **Unsupervised Learning** | No answers; the model finds similarities in data on its own. | User profiling/grouping, anomaly detection |
| **Reinforcement Learning** | Trial and error through environmental feedback to seek rewards. | Autonomous driving, AlphaGo |