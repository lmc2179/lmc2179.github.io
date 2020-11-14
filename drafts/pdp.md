---
layout: post
title: "Partial dependence plots are a simple way to make black-box models easy to understand"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: jellybeans.png
---

*A commonly cited drawback of black-box Machine Learning or nonparametric models is that they're hard to interpret. Sometimes, analysts are even willing to use a model that fits the data poorly because the model is easy to interpret. However, we can often produce clear interpretations of complex models by constructing Partial Dependence Plots. These plots are model-agnostic, easy to implement in Python, and have a natural interpretation even for non-experts. They're awesome tools, and you should use them all the time to understand the relationships you're modeling, even when the model that best fits the data is really complex.*

# Modeling complex relationships often requires complex models

# But black-box models can make it hard to understand the effect of a single feature

I've met a number of smart, skilled analysts who at this point will throw up their hands and just fit a model that they know is not very good, but has a clear interpretation. This is really unfortunate, and generally not necessary.

# An example: ???

Research question: Relationship between NOX and housing prices

```python
from sklearn.datasets import load_boston
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

boston_data = load_boston()
X = pd.DataFrame(boston_data['data'], columns=boston_data['feature_names'])
y = boston_data['target']

# Compare linear regression and random forest regressor; uninterpretable RF fits the data better
mse_linear_model = -cross_val_score(LinearRegression(), X, y, cv=100, scoring='neg_root_mean_squared_error')
mse_rf_model = -cross_val_score(RandomForestRegressor(n_estimators=100), X, y, cv=100, scoring='neg_root_mean_squared_error')
mse_reduction = mse_rf_model - mse_linear_model # Log-log?
```

# Option 1: Make a scatter plot and ignore the other variables

# Option 2: Build a parametric model with a clear interpretation, like a linear model

# Option 3: Build a complex model and use a partial dependence plot

# "Significance tests" and Confidence intervals for black-box models with bootstrapping

# When does the PDP represent a causal relationship?

Note the assumptions from the paper

# Epilogue: So are machine learning models and PDPs the solution to every modeling problem?

No, the statistical theory around regression models is often the right solution but you should be prepared to realize when it's not the only solution

https://christophm.github.io/interpretable-ml-book/pdp.html#fnref28
https://scikit-learn.org/stable/modules/partial_dependence.html
https://web.stanford.edu/~hastie/Papers/pdp_zhao.pdf
