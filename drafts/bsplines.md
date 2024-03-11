---
layout: post
title: "Gams"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: bad_day.png
---

https://replit.com/@srs_moonlight/GAM

# GAMs offer a lot of the advantages of Linear Models with a more flexible functional form

## How I use GAMs

My workflow: linear modeling intuition, then black-box

Intepretability of linear model with non-linear relationships

Linear models have a lot of advantages: Fast, interpretable, statistics

But casual inspection of real life shows that nonlinear relationships are everywhere

You could do black box + PDP, but that's annoying

GAMs are a middle ground, allowing analysis of non-linear functional forms with the easy of linear models

## How GAMs work

Functional form is

$y \sim \alpha + \sum_{i=1}^{k} f_i (x_i) + \epsilon$

In theory, $f_i$ can be and $f : \mathbb{R} \rightarrow \mathbb{R}$

https://pdodds.w3.uvm.edu/files/papers/others/1986/hastie1986a.pdf

Cute idea of backfitting (shalizi)

In practice, splines - show the b-spline basis over the x-axis

## Quickstart/example

```python
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
from statsmodels.gam.api import GLMGam, BSplines

df = fetch_california_housing(as_frame=True)['data']
df['MedHouseVal'] = fetch_california_housing().target

x_col_list = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
x_spline_df_list = [10, 10, 10, 10, 10, 10, 10, 10]
x_spline_degree_list = [3, 3, 3, 3, 3, 3, 3, 3]

X_train_raw = df[x_col_list]

bs = BSplines(X_train_raw, df=x_spline_df_list, degree=x_spline_degree_list)

gam_bs = GLMGam.from_formula('MedHouseVal ~ HouseAge + AveRooms + AveBedrms + Population + AveOccup + Latitude + Longitude', data=df, smoother=bs)

result = gam_bs.fit()
```

# Interpretation

## Summary

`result.summary()`

## Partial plot

`result.plot_partial(0, cpr=True, plot_se=True)`

## P-value on f_i

`print(result.test_significance(0))`

# Fit + CV

## Regularized form

alpha

## Random hyperparameter search

Hyperparams:

* Spline df
* Spline degree
* Alpha

## AIC-based search

```python
# AIC Search

x_col_list = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

aic_results = []

n_runs = 200

for i in tqdm(range(n_runs)):
    x_spline_df_list = list(np.random.randint(4, 12+1, size=len(x_col_list)))
    x_spline_degree_list = [3, 3, 3, 3, 3, 3, 3, 3]
    
    X_train_raw = df[x_col_list]
    
    bs = BSplines(X_train_raw, df=x_spline_df_list, degree=x_spline_degree_list)
    
    gam_bs = GLMGam.from_formula('MedHouseVal ~ HouseAge + AveRooms + AveBedrms + Population + AveOccup + Latitude + Longitude', data=df, smoother=bs)
    
    result = gam_bs.fit()
    aic_results.append(x_spline_df_list + [result.aic])

summary = pd.DataFrame(aic_results, columns=x_col_list + ['AIC'])

from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels import api as sm

sns.regplot(data=summary, x='AveRooms', y='AIC', x_bins=np.arange(4, 12+1)) # Really matters

result.plot_partial(0, cpr=True, plot_se=True)
```

## CV-based search

# Appendix: Intuition behind the AIC

Shalizi

------------------------------------------------------------------------------------------------------------------------------------------------------------

# Previous version

# Bsplines rock and should be your go-to method for modeling smooth nonlinearities - Bsplines and GAMs in Python

https://replit.com/@srs_moonlight/bsplines#main.py

How they work - Basis expansion and splines; Degree of spline, number of knots; start with low order and increase

Example: Non-linear relationship of NOX and Price in the Boston housing data

How to use them in Python

Graphing them with a PDP; [partregress](https://www.statsmodels.org/stable/generated/statsmodels.graphics.regressionplots.plot_partregress.html#statsmodels.graphics.regressionplots.plot_partregress) doesn't work for non-linear stuff

Looking at CIs of the PDP

Maybe do it using sklearn's [PDP tools](https://scikit-learn.org/stable/modules/partial_dependence.html) and a formula transformer

Or, us a GAM! https://www.statsmodels.org/stable/gam.html . Use one standard error rule

print(result.test_significance(0))

https://pdodds.w3.uvm.edu/files/papers/others/1986/hastie1986a.pdf

```python
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
from statsmodels.gam.api import GLMGam, BSplines

df = fetch_california_housing(as_frame=True)['data']
df['MedHouseVal'] = fetch_california_housing().target

x_col_list = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
x_spline_df_list = [10, 10, 10, 10, 10, 10, 10, 10]
x_spline_degree_list = [3, 3, 3, 3, 3, 3, 3, 3]

X_train_raw = df[x_col_list]

bs = BSplines(X_train_raw, df=x_spline_df_list, degree=x_spline_degree_list)

gam_bs = GLMGam.from_formula('MedHouseVal ~ HouseAge + AveRooms + AveBedrms + Population + AveOccup + Latitude + Longitude', data=df, smoother=bs)

result = gam_bs.fit()

result.plot_partial(0, cpr=True, plot_se=True)

print(result.test_significance(0))
```

GAM degrees of freedom: two (spline size, alpha) per real term. Do a random search (Bergstrom paper) and look at AIC, then CV. Compare results



https://en.wikipedia.org/wiki/Akaike_information_criterion

Seychelles diagram

Preidciton intervals?

# Appendix

```python
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels.api import formula as smf
import pandas as pd
from statsmodels.graphics.regressionplots import plot_partregress

n = 100
x = np.linspace(0, 7, n)
y = np.sin(x) + x + np.random.normal(0, 1, size=n)
data = pd.DataFrame({'x': x, 'y': y})

spline_df = 10

for spline_degree in np.arange(0, 4):
  model_spec = 'y ~ x + bs(x, df={0}, degree={1})'.format(spline_df, spline_degree)
  model = smf.ols(model_spec, data)
  result = model.fit()
  y_hat = result.fittedvalues
  
  pred_results = result.get_prediction(data)
  y_hat_se = pred_results.se_mean
  y_hat_low = y_hat - 2 * y_hat_se
  y_hat_high = y_hat + 2 * y_hat_se
  
  y_obs_se = pred_results.se_obs
  y_hat_obs_low = y_hat - 2 * y_obs_se
  y_hat_obs_high = y_hat + 2 * y_obs_se
  
  plt.scatter(x, y)
  plt.plot(x, np.sin(x) + x)
  plt.plot(x, y_hat)
  plt.fill_between(x, y_hat_low, y_hat_high, color='grey', alpha=.5)
  plt.plot(x, y_hat_obs_high)
  plt.plot(x, y_hat_obs_low)
  plt.title(model_spec)
  plt.show()
  ```
