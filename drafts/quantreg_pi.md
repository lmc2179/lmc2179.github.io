---
layout: post
title: "Flexible prediction intervals: Quantile Regression in Python"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: quantile_friend.png
---

## Most useful forecasts include a range of likely outcomes

It's generally good to try and guess what the future will look like, so we can plan accordingly. How much will our new inventory cost? How many users will show up tomorrow? How much raw material will I need to buy? The first instinct we have is usual to look at historical averages; we know the average price of widgets, the average number of users, etc. If we're feeling extra fancy, we might build a model, like a linear regression, but this is also an average; a conditional average based on some covariates. Most out-of-the-box machine learning models are the same, giving us a prediction that is correct on average.

However, answering these questions with a single number, like an average, is a little dangerous. The actual cost will usually not be exactly the average; it will be somewhat higher or lower. How much higher? How much lower? If we could answer this question with a range of values, we could prepare appropriately for the worst and best case scenarios. It's good to know our resource requirements for the average case; it's better to also know the worst case (even if we don't expect the worst to actually happen, if total catastrophe is plausible it will change our plans).

As is so often the case, it's useful to consider a specific example. Let's imagine a seasonal product; to pick one totally at random, imagine the inventory planning of a luxury sunglasses brand for cats. Purrberry needs to make summer sales projections for inventory allocation across its various brick-and-mortar locations where it's sales happen.

You go to your data warehouse, and pull last year's data on each location's pre-summer sales (X-axis) and summer sales (Y-axis):

```python
from matplotlib import pyplot as plt
import seaborn as sns

plt.scatter(df['off_season_revenue'], df['on_season_revenue'])
plt.xlabel('Off season revenue at location')
plt.ylabel('On season revenue at location')
plt.title('Comparison between on and off season revenue at store locations')
plt.show()
```
![Scatterplot](https://raw.githubusercontent.com/lmc2179/lmc2179.github.io/master/assets/img/quantreg_pi/Figure_1.png)

We can read off a few things here straight away:
* A location with high off-season sales will also have high summer sales; X and Y are positively correlated.
* The outcomes are more certain for the stores with the highest off-season sales; the variance of Y increases with X.
* On the high end, outlier results are more likely to be extra high sales numbers instead of extra low; the noise is asymmetric, and positively skewed.

After this first peek at the data, you might reach for that old standby, Linear Regression.

## Our usual tool, OLS, doesn't always handle this well

Regression afficionados will recall that our [trusty OLS model allows us to compute prediction intervals](https://lmc2179.github.io/posts/confidence_prediction.html), so we'll try that first.

Recall that the OLS model is

$y ~ \alpha + \beta x + N(0, \sigma)$

Where $\alpha$ is the intercept, $\beta$ is the slope, and $\sigma$ is the standard deviation of the residual distribution. Under this model, we expect that observations of $y$ are normally distributed around $\alpha + \beta x$, with a standard deviation of $\sigma$. We estimate $\alpha$ and $\beta$ the usual way, and look at the observed residual variance to estimate $\sigma$, and we can use the [familiar properties of the normal distribution](https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule) to create prediction intervals.

```python
from statsmodels.api import formula as smf

ols_model = smf.ols('on_season_revenue ~ off_season_revenue', df).fit()
predictions = ols_model.predict(df)
resid_sd = np.std(ols_model.resid)

high, low = predictions + 1.645 * resid_sd, predictions - 1.645 * resid_sd

plt.scatter(df['off_season_revenue'], df['on_season_revenue'])
plt.plot(df['off_season_revenue'], high, label='OLS 90% high PI')
plt.plot(df['off_season_revenue'], predictions, label='OLS prediction')
plt.plot(df['off_season_revenue'], low, label='OLS 90% low PI')
plt.legend()
plt.xlabel('Off season revenue at location')
plt.ylabel('On season revenue at location')
plt.title('OLS prediction intervals')
plt.show()
```

![OLS plot](https://raw.githubusercontent.com/lmc2179/lmc2179.github.io/master/assets/img/quantreg_pi/Figure_2.png)

Hm. Well, this isn't terrible - it looks like the 90% prediction intervals do contain the majority of observations. However, it also looks pretty suspect; on the left side of the plot the PIs seem too broad, and on the right side they seem a little too narrow.

This is because the PIs are the same width everywhere, since we assumed that the variance of the residuals is the same everywhere. But from this plot, we can see that's not true; the variance increases as we increase X. These two situations (constant vs non-constant variance) have the totally outrageous names [homoskedasticity and heteroskedasticity](https://en.wikipedia.org/wiki/Homoscedasticity_and_heteroscedasticity). OLS assumes homoskedasticity, but we actually have heteroskedasticity. If we want to make predictions that match the data we see, and OLS model won't quite cut it.

NB: A choice sometimes recommended in a situation like this is to perform a log transformation, but [we've seen before](https://lmc2179.github.io/posts/multiplicative.html) that logarithms aren't a panacea when it comes to heteroskedasticity, so we'll skip that one.

## The idea: create prediction intervals based on the conditional quantiles

We really want to answer a question like: "For a store with $x$ pre-summer sales, where will (say) 90% of the summer sales be?". We want to know how the bounds of the distribution, the highest and lowest plausible observations, change with the pre-summer sales numbers. If we weren't considering an input, we might look at the 5% and 95% [quantiles](https://en.wikipedia.org/wiki/Quantile) of the data to answer that question.

Conditional quantile is kind of like conditional mean from ols

Start with OLS, change loss to get conditional quantile

$\mathbb{E}[y \mid x]$

$Median[y \mid x]$ https://en.wikipedia.org/wiki/Least_absolute_deviations

$\mathbb{Q}[y \mid x]$


https://en.wikipedia.org/wiki/Quantile_regression#Conditional_quantile_and_quantile_regression

the whimsically name [pinball loss function](https://www.lokad.com/pinball-loss-function-definition)

# Quantile regression in action

## Fitting the model

it's just like fitting an OLS model in R or Python

```python
high_model = smf.quantreg('on_season_revenue ~ off_season_revenue', df).fit(q=.95)
mid_model = smf.quantreg('on_season_revenue ~ off_season_revenue', df).fit(q=.5)
low_model = smf.quantreg('on_season_revenue ~ off_season_revenue', df).fit(q=.05)

plt.scatter(df['off_season_revenue'], df['on_season_revenue'])
plt.plot(df['off_season_revenue'], high_model.predict(df), label='95% Quantile')
plt.plot(df['off_season_revenue'], mid_model.predict(df), label='50% Quantile (Median)')
plt.plot(df['off_season_revenue'], low_model.predict(df), label='5% Quantile')
plt.legend()
plt.xlabel('Off season revenue at location')
plt.ylabel('On season revenue at location')
plt.title('Quantile Regression prediction intervals')
plt.show()
```

![Quantreg scatterplot](https://raw.githubusercontent.com/lmc2179/lmc2179.github.io/master/assets/img/quantreg_pi/Figure_3.png)



Evidence of heteroskedasticity? differing slopes of high and low

Evidence of asymmetry? high - mid == mid - low

## Checking the model

Coverage is the percentage of data points which fall into the predicted range

```python
from scipy.stats import sem
covered = (df['on_season_revenue'] >= low_model.predict(df)) & (df['on_season_revenue'] <= high_model.predict(df))
print('In-sample coverage rate: ', np.average(covered))
print('Coverage SE: ', sem(covered))
```
```
In-sample coverage rate:  0.896
Coverage SE:  0.019345100974843932
```

Coverage CI using the SEM

Could use this as a model selection metric if our goal is to find a model specification that maximizes predictive power

we should see that the coverage is good across the support: Coverage plot

consider extending to percentile uniformity

```python
sns.regplot(df['off_season_revenue'], covered, x_bins=4)
plt.axhline(.9, linestyle='dotted', color='black')
plt.title('Coverage by revenue group')
plt.xlabel('Off season revenue at location')
plt.ylabel('Coverage')
plt.show()
```

![Coverage plot](https://raw.githubusercontent.com/lmc2179/lmc2179.github.io/master/assets/img/quantreg_pi/Figure_4.png)

What other models might we have considered? splines or other non-linearities

# Some other perspectives on quantile regression and prediction intervals

Other uses of quantile regression: seeing distributional impact when there are many covariates

Other ways of doing PIs: link to shalizi

# Appendix: How the data was generated

The feline fashion visionaries at Purrberry are, regrettably, entirely fictional for the time being. The data from this example was generated using the below code, which creates skew normal distributed noise:

```python
import numpy as np
from scipy.stats import skewnorm
import pandas as pd

n = 250
x = np.linspace(.1, 1, n)
gen = skewnorm(np.arange(len(x))+.01, scale=x)
gen.random_state = np.random.Generator(np.random.PCG64(abs(hash('predictions'))))
y = 1 + x + gen.rvs()

df = pd.DataFrame({'off_season_revenue': x, 'on_season_revenue': y})
```
