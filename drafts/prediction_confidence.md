---
layout: post
title: "Understanding the difference between prediction and confidence intervals in Python"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: prediction_confidence.png
---

# Prediction and confidence intervals are a common source of confusion

# Example: An OLS regression model

```python
from scipy.stats import norm
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels.api import formula as smf
import pandas as pd
import numpy as np

n = 100
a = 1
b = 5
s = 1

df = pd.DataFrame({'x': np.linspace(0, 1, n), 'y':a + b*np.linspace(0, 1, n) + norm(0, s).rvs(n)})

model = smf.ols('y ~ x', df)
results = model.fit()

predictions = results.get_prediction(df).summary_frame()

plt.fill_between(df['x'], predictions['obs_ci_lower'], predictions['obs_ci_upper'], alpha=.1, label='Prediction interval')
plt.fill_between(df['x'], predictions['mean_ci_lower'], predictions['mean_ci_upper'], alpha=.5, label='Confidence interval')
plt.scatter(df['x'], df['y'], label='Observed', marker='x', color='black')
plt.plot(df['x'], predictions['mean'], label='Point predicton')
plt.legend()
plt.show()
```

-Picture

# Confidence intervals around the parameters

# Confidence intervals around the predictions
# Prediction intervals
# What about a non-OLS model?

Logit: https://stackoverflow.com/questions/47414842/confidence-interval-of-probability-prediction-from-logistic-regression-statsmode

# What if my model is misspecified?
