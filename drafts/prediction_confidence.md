---
layout: post
title: "Understanding the difference between prediction and confidence intervals for linear models in Python"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: cat_sunglasses.jpg
---

*The difference between prediction and confidence intervals is often confusing to newcomers, as the distinction between them is often described in statistics jargon that's hard to follow intuitively. This is unfortunate, because they are useful concepts, and worth exploring for practitioners, even those who don't much care for statistics jargon. This post will walk through some ways of thinking about these important concepts, and demonstrate how we can calculate them for OLS and Logit models in Python. Plus, cats in sunglasses.*

# Example: An OLS regression model with one independent variable

As always, it's useful to get some intuition by looking at an example, so we'll start with one. Let's say that you, a serial innovator and entrepreneur, have recently started designing your very own line of that most coveted feline fashion accessory, [little tiny sunglasses for cats](https://www.amazon.com/Coolrunner-Sunglasses-Classic-Circular-Fashion/dp/B07748RLF5). After the initial enormous rush of sales subsides, you sit back to consider the strategic positioning of your business, and you realize that it's quite likely that your business is a seasonal one. Specifically, you're pretty sure that cats are most likely to purchase sunglasses during the warm months, when they're most likely to find themselves outdoors in need of a snappy accessory.

You collect some data on the daily temperature and dollars of sales in the region where you do most of your sales, and you plot them:

[Scatter plot]

So far, so good. It looks like the two are correlated positively. You fit yourself a line

[Line]

The line has a positive slope, as we expect. Of course, this is only a sample of daily temperatures, and we 

[CI of mean]

This tells us something about the uncertainty

[Full plot with all the stuff]

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

df = pd.DataFrame({'temperature': np.linspace(0, 1, n), 'sales':a + b*np.linspace(0, 1, n) + norm(0, s).rvs(n)})

model = smf.ols('sales ~ temperature', df)
results = model.fit()

predictions = results.get_prediction(df).summary_frame()

plt.fill_between(df['temperature'], predictions['obs_ci_lower'], predictions['obs_ci_upper'], alpha=.1, label='Prediction interval')
plt.fill_between(df['temperature'], predictions['mean_ci_lower'], predictions['mean_ci_upper'], alpha=.5, label='Confidence interval')
plt.scatter(df['temperature'], df['sales'], label='Observed', marker='x', color='black')
plt.plot(df['temperature'], predictions['mean'], label='Point predicton')
plt.legend()
plt.show()
```

We've got quite a dense plot now - let's take some time and recap all the elements we've added before we tackle them in detail:

- The observed datapoints are `(temperature, sales revenue)` pairs.
- The regression line tells us what the average revenue is as the temperature varies in our dataset. Here, we've assumed that the revenue varies linearly with the temperature. The regression line answers the question: "If we know the temperature, what is our single best guess about the average level of sales we expect to see?"
- The confidence interval tells us the range of the average revenue at a given temperature. It answers the question: "If we know the temperature, what is our uncertainty around the average level of sales?"
- The prediction interval tells us the range where the oberved revenue on an actual day is likely to fall at a given temperature. It answers the question: "If we know the temperature, what actual range of sales might we see on a given day?".

# Recap: What is the OLS model doing?

In this example, we had two variables: $temperature$ and $sales$. We want to know how $sales$ changes as $temperature$ varies. X and y

We assume that we can write $y$ as a function of $X$, something like

$$ y = f(X) + \epsilon $$

The idea is that $f(X)$ tells us how the average of $y$ changes with $X$, and $\epsilon$ is a normally-distributed noise term from all the variables we didn't choose to put in our model. If we knew the exact form of $f(X)$, we could always compute the expected of $y$ given $X$, namely $\mathbb{E}[y \mid X] = f(X)$.

There are many choices of $f$ we could pick, but for convenience we often assume that $y$ is a linear function of $X$.

So it has unknown parameters beta

OLS estimates beta

The prediction for a given X is ... but actual observations will be "near" it

Gives us $\mathbb{E} [y \mid X]$ plus some gaussian noise

# Confidence intervals around the predictions

## What does the CI for the predictions represent?

CI around $\mathbb{E} [y \mid X]$

## Where does it come from?

Section 8.1 of http://www.stat.cmu.edu/~cshalizi/TALR/TALR.pdf

# Prediction intervals

## What does the prediction interval represent?

Range of observed $y \mid X$

## Where does it come from?

Section 8.2 of http://www.stat.cmu.edu/~cshalizi/TALR/TALR.pdf

## We can check to see if the prediction intervals have the expected coverage

Check probability of covarage

Plot probability of coverage vs your favorite variable

# One more thing: Confidence intervals around the parameters

## What does the CI for the parameters represent?

## Where does it come from?

12.4.2 of http://www.stat.cmu.edu/~cshalizi/TALR/TALR.pdf

# What about a different GLM, like a logit model?

Logit: https://stackoverflow.com/questions/47414842/confidence-interval-of-probability-prediction-from-logistic-regression-statsmode

Brief mention of delta method, link to p. 69 of http://www.stat.cmu.edu/~cshalizi/TALR/TALR.pdf, see https://en.wikipedia.org/wiki/Delta_method#Multivariate_delta_method
Mention the bootstrap for non-asymptotic stuff

Works out of the box with a GLM for logit models


## Can a logit model have a prediction interval?

Yes but it's not useful

# Appendix: Imports
