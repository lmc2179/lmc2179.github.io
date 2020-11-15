---
layout: post
title: "Partial dependence plots are a simple way to make black-box models easy to understand"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: Shiraz.png
---

*A commonly cited drawback of black-box Machine Learning or nonparametric models is that they're hard to interpret. Sometimes, analysts are even willing to use a model that fits the data poorly because the model is easy to interpret. However, we can often produce clear interpretations of complex models by constructing Partial Dependence Plots. These plots are model-agnostic, easy to implement in Python, and have a natural interpretation even for non-experts. They're awesome tools, and you should use them all the time to understand the relationships you're modeling, even when the model that best fits the data is really complex.*

# Modeling complex relationships often requires complex models

If you 

https://en.wikipedia.org/wiki/Cross-validation_(statistics)

Luckily, things have come a long way

# But black-box models can make it hard to understand the effect of a single feature

However, we often want more than _just_ a model that makes good predictions. We frequently want to use our models to expand our intuition about the relationships between variables. And more than that, most consumers of a model are skeptical, intelligent people, who want to understand how the model works before they're willing to trust it.

At this point, though, you're caught in an impasse. If you fit a

I've met a number of smart, skilled analysts who at this point will throw up their hands and just fit a model that they know is not very good, but has a clear interpretation. This is understandable, since an approximate solution is better than no solution - but it's not necessary, as we can produce much better approximations which are still interpretable.

# An example: The relationship between air quality and housing prices

We'll introduce a short example here which we'll revisit a few times. This example involves a straightforward question and small data set, but relationships between variables that are non-linear.

Research question: Relationship between NOX and median house prices

https://scikit-learn.org/stable/datasets/index.html#boston-dataset

```python
from sklearn.datasets import load_boston
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.inspection import partial_dependence
from sklearn.utils import resample

boston_data = load_boston()
X = pd.DataFrame(boston_data['data'], columns=boston_data['feature_names'])
y = boston_data['target']

# Compare linear regression and random forest regressor; uninterpretable RF fits the data better
mse_linear_model = -cross_val_score(LinearRegression(), X, y, cv=100, scoring='neg_root_mean_squared_error')
mse_rf_model = -cross_val_score(RandomForestRegressor(n_estimators=100), X, y, cv=100, scoring='neg_root_mean_squared_error')
mse_reduction = mse_rf_model - mse_linear_model
# CI on mse_reduction - demonstrates that the rf model has better predictive power
```

# Option 1: Make a scatter plot and ignore the other variables

Usually, our first instinct is to make a scatter plot

```python
sns.regplot(X['NOX'], y)
plt.show()
```

This is a perfectly good start, and often worth doing. However, this scatter plot alone doesn't actually answer our question because it doesn't give us the unique effect of NOX

# Option 2: Build a parametric model with a clear interpretation, like a linear model

We can think of the last section as a very simple model, but we know this was an oversimplification

At this point, we can expand our model to be more realistic by including the other variables that we believe affect home prices

```python
sm.OLS(y, X).fit().summary()
```

Splendid

We can visualize this relationship with a partial regression plot
https://en.wikipedia.org/wiki/Partial_regression_plot
https://www.statsmodels.org/stable/generated/statsmodels.graphics.regressionplots.plot_partregress.html#statsmodels.graphics.regressionplots.plot_partregress

```python
plot_partregress(y, X['NOX'], X.drop('NOX', axis=1), obs_labels=False)
plt.show()
```

So more NOX is correlated with lower prices, even when we account for the other variables

We assumed the relationship was linear

```python
sns.regplot(X['NOX'], 
            sm.OLS(y, X).fit().resid, 
            lowess=True, 
            scatter_kws={'alpha': .1})
plt.axhline(0, linestyle='dotted')
plt.show()
```

Oh no

If we are working with linearity assumptions, this is where we stop - we include the above plot in our report, with an asterisk that the relationship isn't exactly linear

We might expand our model to consider nonlinear and interaction terms, fair enough

In the next section, we'll introduce a tool that lets us get past this and directly interpret our best-fitting model

# Option 3: Build a complex model and use a partial dependence plot

So the model with the best out-of-sample performance is a random forest, okay fair enough

But under the hood a random forest includes a whole bunch of decisions trees combined in an opaque way

We can look at the information gain, that's useful but it just tells us "this variable is important" 

"What happens to NOX when all other variables are held constant"

1. Set all NOX to some value, leaving all other variables the same
2. Predict 
3. Average
Repeat

```python
rf_model = RandomForestRegressor(n_estimators=100).fit(X, y)

nox_values = np.linspace(np.min(X['NOX']), np.max(X['NOX']))

pdp_values = []
for n in nox_values:
  X_pdp = X.copy()
  X_pdp['NOX'] = n
  pdp_values.append(np.mean(rf_model.predict(X_pdp)))

plt.plot(nox_values, pdp_values)
plt.show()
```

Oh wow look at that non-linearity isn't that interesting

That right there is the PDP - easy to code, easy to understand, though it might take a lot of computing power

# Confidence intervals for PDPs with bootstrapping

One thing we like about linear models is that we can compute some significance

Standard errors in the LR model come from the T-distribution

https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.conf_int.html
https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.pvalues.html
https://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/18/lecture-18.pdf

Bootstrapping

```python
n_bootstrap = 100

nox_values = np.linspace(np.min(X['NOX']), np.max(X['NOX']))

for _ in range(n_bootstrap): # This should probably be bands
    X_boot, y_boot = resample(X, y)
    rf_model_boot = RandomForestRegressor(n_estimators=100).fit(X_boot, y_boot)
    
    pdp_values = []
    for n in nox_values:
        X_pdp = X_boot.copy()
        X_pdp['NOX'] = n
        pdp_values.append(np.mean(rf_model.predict(X_pdp)))
    plt.plot(nox_values, pdp_values, color='blue')

plt.show()
```

# When does the PDP represent a causal relationship?

This section uses a bit of language from CI

Note the assumptions from the paper

- NOX is not a cause of any other predictor variables
- The other predictor variables block all back-door paths between NOX and house price

THESE ARE ASSUMPTIONS AND WE CAN'T CHECK THEM

https://web.stanford.edu/~hastie/Papers/pdp_zhao.pdf

# Epilogue: So are machine learning models and PDPs the solution to every modeling problem?

No, the statistical theory around regression models is often the right solution but you should be prepared to realize when it's not the only solution

The computational costs might be quite large

I use a bootstrapping solution but we could have done something else like https://jmlr.org/papers/volume15/wager14a/wager14a.pdf

# Some further reading

https://christophm.github.io/interpretable-ml-book/pdp.html
https://scikit-learn.org/stable/modules/partial_dependence.html
https://web.stanford.edu/~hastie/Papers/pdp_zhao.pdf
