---
layout: post
title: "Understanding the difference between prediction and confidence intervals for linear models in Python"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: cat_sunglasses.jpg
---

*The difference between prediction and confidence intervals is often confusing to newcomers, as the distinction between them is often described in statistics jargon that's hard to follow intuitively. This is unfortunate, because they are useful concepts, and worth exploring for practitioners, even those who don't much care for statistics jargon. This post will walk through some ways of thinking about these important concepts, and demonstrate how we can calculate them for OLS and Logit models in Python.*

https://www.amazon.com/Coolrunner-Sunglasses-Classic-Circular-Fashion/dp/B07748RLF5/ref=asc_df_B07748RLF5/?tag=hyprod-20&linkCode=df0&hvadid=241996956146&hvpos=&hvnetw=g&hvrand=7196555194832522535&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=1019973&hvtargid=pla-445582810531&psc=1

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

# Recap: What is the OLS model doing?

# Confidence intervals around the parameters

## What does the CI for the parameters represent?

## Where does it come from?

12.4.2 of http://www.stat.cmu.edu/~cshalizi/TALR/TALR.pdf

# Confidence intervals around the predictions

## What does the CI for the predictions represent?

CI around E[y|X]

## Where does it come from?

Section 8.1 of http://www.stat.cmu.edu/~cshalizi/TALR/TALR.pdf

# Prediction intervals

## What does the prediction interval represent?

Range of observed y | X

## Where does it come from?

Section 8.2 of http://www.stat.cmu.edu/~cshalizi/TALR/TALR.pdf

## We can check to see if the prediction intervals have the expected coverage

# What about a different GLM, like a logit model?

Logit: https://stackoverflow.com/questions/47414842/confidence-interval-of-probability-prediction-from-logistic-regression-statsmode

## The asymptotic version for the CI of the predictions: The delta method

Brief mention of delta method, link to p. 69 of http://www.stat.cmu.edu/~cshalizi/TALR/TALR.pdf, see https://en.wikipedia.org/wiki/Delta_method#Multivariate_delta_method

Works out of the box with a GLM for logit models

## An alternative version: The bootstrap

?


## Can a logit model have a prediction interval?

Yes but it's not useful



# Appendix: Imports
