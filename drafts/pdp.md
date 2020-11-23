---
layout: post
title: "Partial dependence plots are a simple way to make black-box models easy to understand"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: Shiraz.png
---

*A commonly cited drawback of black-box Machine Learning or nonparametric models is that they're hard to interpret. Sometimes, analysts are even willing to use a model that fits the data poorly because the model is easy to interpret. However, we can often produce clear interpretations of complex models by constructing Partial Dependence Plots. These plots are model-agnostic, easy to implement in Python, and have a natural interpretation even for non-experts. They're awesome tools, and you should use them all the time to understand the relationships you're modeling, even when the model that best fits the data is really complex.*

# If we want to understand complex relationships, we need complex models

We frequently model the relationships between a set of variables and an outcome of interest by building a model. This might be so we can make predictions about unseen outcomes, or so we can build a theory of how the variables affect the outcome, or simply to describe the observed relationships between the variables. Whatever our goal, we collect a bunch of examples, then infer a model that relates the inputs to the outcome.

A data analyst with access to R or Python has a ton of powerful modeling tools at their disposal. With a single line of scikit-learn, they can often produce a model with a substantial predictive power. The last 70 or so years of machine learning and nonparametric modeling research allows us to produce models that make good predictions without much explicit feature engineering, which automatically find interactions or nonlinearities, and so on. A common workflow is to consider a set of models which are a priori plausible, and select the best model (or the best few) using a procedure based on [cross validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)). You might simply pick the model with the best out-of-sample error, or perhaps one that is both [parsimonious and makes good predictions](https://lmc2179.github.io/posts/cvci.html). The result: after you've done all this sklearning and grid searching and gradient descending and whatever else, you've got a model that accurately predicts your data. Because of all this fancy business, the resulting model might be complex - it might be a random forest with a thousand trees, or an boosted collection of learners, or a neural network with a bunch of hidden layers. 

# But black-box models can make it hard to understand the effect of a single feature

In many real-world situations, we can use all these fancy libraries to find a black-box model that fits our data well. However, we often want more than _just_ a model that makes good predictions. We frequently want to use our models to expand our intuition about the relationships between variables. And more than that, most consumers of a model are skeptical, intelligent people, who want to understand how the model works before they're willing to trust it. We may even want to use our model to understand how to build interventions or causal relationships.

What if the model that best fits your data is a complex black-box model, but you also want to do some intuition-building? At this point, though, you're caught in an impasse. If you fit a simple model which fits the data badly, you'll have a poor approximation with high interpretability. If you fit a black-box model which approximates the relationship well out of sample, you may find yourself unable to understand how your model works, and build any useful intuitive knowledge. 

I've met a number of smart, skilled analysts who at this point will throw up their hands and just fit a model that they know is not very good, but has a clear interpretation. This is understandable, since an approximate solution is better than no solution - but it's not necessary, as it turns out even black-box approximations which are still interpretable if we think about it the right way. We'll look at a specific example of this, and walk through how to do it in Python.

# An example: The relationship between air quality and housing prices

We'll introduce a short example here which we'll revisit from a few perspectives. This example involves a straightforward question and small data set, but relationships between variables that are non-linear and possible interactions. The data is the classic [Boston Housing dataset](
https://scikit-learn.org/stable/datasets/index.html#boston-dataset), available in sklearn. This data originally came from an investigation of the relationship between air quality, as measured by nitric oxide ("NOX") concentration, and median house price. The data includes data from a number of Boston neighborhoods in the 1970s, and includes their measured NOX, median house price, and other variables indicating factors that might affect house price (like the business and demographic makeup of the area). We'll ask the research question: *What is the relationship between NOX and house price?* We'll break that down into two further questions: 

- *All else being equal, do changes in NOX correlate with changes in house price in this data set?* 
- *Could we say that NOX changes cause changes in median house price?*

Note that these are two different questions! The first one is about correlation, and we can answer it just with the data at hand. The second one is a much more tricky question, and we won't answer it definitively here; however, we'll talk about what we _would_ need to convincingly answer that question. We'll mostly focus on the first question, but we'll talk about the second in our last section.

Let's write a bit of code to grab the data and start down the road to answering these questions. We'll begin by importing a bunch of things:

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
from scipy.stats import sem
import numpy as np
from statsmodels.graphics.regressionplots import plot_partregress
```

We load the data from sklearn:

```python
boston_data = load_boston()
X = pd.DataFrame(boston_data['data'], columns=boston_data['feature_names'])
y = boston_data['target']
```

If we knew the true relationship between all the `X` variables and the `y` variable, we could answer our questions above. Of course, we don't know the true relationship, so we'll attempt to infer it (as best we can) from the data we've collected. That is, we'll build a model that relates neighborhood attributes (`X`) to home price (`y`) and try to answer our question with that model.

Let's look at a few candidate models, a Linear Regression and a Random Forest:

```python
mse_linear_model = -cross_val_score(LinearRegression(), X, y, cv=100, scoring='neg_root_mean_squared_error')
mse_rf_model = -cross_val_score(RandomForestRegressor(n_estimators=100), X, y, cv=100, scoring='neg_root_mean_squared_error')
mse_reduction = mse_rf_model - mse_linear_model

print('Average MSE for linear regression is {0:.3f}'.format(np.mean(mse_linear_model)))
print('Average MSE for random forest is {0:.3f}'.format(np.mean(mse_rf_model)))
print('Switching to a Random Forest over a linear regression reduces MSE on average by {0:.3f} ± {1:.3f}'.format(np.mean(mse_reduction), 3*sem(mse_reduction)))
```
Output:
```
Average MSE for linear regression is 4.184
Average MSE for random forest is 3.024
Switching to a Random Forest over a linear regression reduces MSE on average by -1.160 ± 0.514
```

We see that the Random Forest model produces better predictive power than the Linear Regression when we look at the out-of-sample RMSE. So far, so good! Perhaps if we dug a little deeper, we'd find a better model - for now, let's assume we're only considering these two. Already, we know something valuable! That is that the random forest does a better job of predicting home prices for neighborhoods it hasn't seen than the linear model does.

# Option 1: Make a scatter plot and ignore the other variables

Let's step back for a moment. Usually, when we are confronted with a "does this variable correlate with that variable" question, we start with a scatterplot. Why not simply make a scatterplot of NOX against median house value? Well, there's nothing stopping us from doing this, so let's do it:

```python
sns.regplot(X['NOX'], y, lowess=True)
plt.ylabel('Median house price')
plt.xlabel('NOX')
plt.title('Scatter plot of NOX vs price with LOWESS fit')
plt.show()
```

![scatter plot of NOX vs price](https://raw.githubusercontent.com/lmc2179/lmc2179.github.io/master/assets/img/pdp/1.png)

This is a perfectly good start, and often worth doing. However, this scatter plot alone doesn't actually answer our question. It _does_ tell us something useful, which is that NOX is negatively correlated with house prices. That is, areas with higher NOX (and thus worse air quality) have a lower house price, on average. But there's a straightforward objection to this finding, which is that our scatterplot ignores the other variables we know about. Perhaps areas with high NOX also have some other attribute which causes lower house prices. We've also plotted the [LOWESS](https://en.wikipedia.org/wiki/Local_regression) fit, giving us some idea of how the average price changes as we look at neighborhoods with different NOX.

So...are we done? All that huffing and puffing so we can answer our question with a scatterplot? NOX is negatively correlated with house price - done.

# Option 2: Build a simple model with a clear interpretation, like a linear model

Not quite. We can think of the last section as a very simple model in which NOX is the sole variable that affects house prices, but we know this was an oversimplification. Specifically, NOX might just be higher in neighborhoods that are undesirable for other reasons, and NOX has nothing to do with it. If this kind of coincidence were really the case, we wouldn't see it in the scatterplot above. We want the _unique_ impact of NOX - that's the "holding all else constant" part of our question above. We can expand our model to be more realistic by including the other variables that we believe affect home prices, hoping to avoid omitted variable bias.

We saw before that a simple linear regression isn't the best model, but perhaps it's good enough for us to learn something from. We'll fit a linear model and look at its summary:

```python
sm.OLS(y, X).fit().summary()
```

This produces the very official-looking regression results:

```
                                 OLS Regression Results
=======================================================================================
Dep. Variable:                      y   R-squared (uncentered):                   0.959
Model:                            OLS   Adj. R-squared (uncentered):              0.958
Method:                 Least Squares   F-statistic:                              891.3
Date:                Mon, 23 Nov 2020   Prob (F-statistic):                        0.00
Time:                        14:12:20   Log-Likelihood:                         -1523.8
No. Observations:                 506   AIC:                                      3074.
Df Residuals:                     493   BIC:                                      3128.
Df Model:                          13
Covariance Type:            nonrobust
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
CRIM          -0.0929      0.034     -2.699      0.007      -0.161      -0.025
ZN             0.0487      0.014      3.382      0.001       0.020       0.077
INDUS         -0.0041      0.064     -0.063      0.950      -0.131       0.123
CHAS           2.8540      0.904      3.157      0.002       1.078       4.630
NOX           -2.8684      3.359     -0.854      0.394      -9.468       3.731
RM             5.9281      0.309     19.178      0.000       5.321       6.535
AGE           -0.0073      0.014     -0.526      0.599      -0.034       0.020
DIS           -0.9685      0.196     -4.951      0.000      -1.353      -0.584
RAD            0.1712      0.067      2.564      0.011       0.040       0.302
TAX           -0.0094      0.004     -2.395      0.017      -0.017      -0.002
PTRATIO       -0.3922      0.110     -3.570      0.000      -0.608      -0.176
B              0.0149      0.003      5.528      0.000       0.010       0.020
LSTAT         -0.4163      0.051     -8.197      0.000      -0.516      -0.317
==============================================================================
Omnibus:                      204.082   Durbin-Watson:                   0.999
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1374.225
Skew:                           1.609   Prob(JB):                    3.90e-299
Kurtosis:                      10.404   Cond. No.                     8.50e+03
==============================================================================
```

Splendid, tells us the sign is negative, and the coefficient is not significant (-9.468, 3.731)

We can visualize this relationship with a partial regression plot
https://en.wikipedia.org/wiki/Partial_regression_plot
https://www.statsmodels.org/stable/generated/statsmodels.graphics.regressionplots.plot_partregress.html#statsmodels.graphics.regressionplots.plot_partregress

```python
plot_partregress(y, X['NOX'], X.drop('NOX', axis=1), obs_labels=False)
plt.axhline(0, linestyle='dotted')
plt.ylim(-2, 2)
plt.show()
```

![partial regression plot of NOX vs price](https://raw.githubusercontent.com/lmc2179/lmc2179.github.io/master/assets/img/pdp/2.png)

So more NOX is correlated with lower prices, even when we account for the other variables. The relationship is not significant, though

Regression is a powerful tool for understanding the unique relationship between many variables and an outcome - there's a reason it's one of the most-used tools in your kit. However, we made some assumptions along the way. We'll interrogate one of those assumptions, which is that the outcome varies linearly with each covariate. We'll check that using a popular [regression diagnostic](https://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/07/lecture-07.pdf), as plot of covariate vs residuals:

```python
sns.regplot(X['NOX'], 
            sm.OLS(y, X).fit().resid, 
            lowess=True, 
            scatter_kws={'alpha': .1})
plt.axhline(0, linestyle='dotted')
plt.title('Residual diagnostic')
plt.xlabel('NOX')
plt.ylabel('Predicted - actual')
plt.ylim(-5, 5)
plt.show()
```

![residual diagnostic](https://raw.githubusercontent.com/lmc2179/lmc2179.github.io/master/assets/img/pdp/3.png)

This is 

If we are working with linearity assumptions, this is where we stop - we include the above plot in our report, with an asterisk that the relationship isn't exactly linear but we have a linear approximation to it

We might expand our model to consider nonlinear and interaction terms, fair enough

In the next section, we'll introduce a tool that lets us get past this and directly interpret our best-fitting model

# Option 3: Build a more complex model and use a partial dependence plot

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
plt.ylabel('Predicted house price')
plt.xlabel('NOX')
plt.title('Partial dependence plot for NOX vs Price for Random Forest')
plt.show()
```


![partial dependence plot](https://raw.githubusercontent.com/lmc2179/lmc2179.github.io/master/assets/img/pdp/4.png)

Oh wow look at that non-linearity isn't that interesting

That right there is the PDP - easy to code, easy to understand, though it might take a lot of computing power

# Confidence intervals for PDPs with bootstrapping

One thing we like about linear models is that we can compute some significance

Standard errors in the LR model come from the T-distribution

https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.conf_int.html
https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.pvalues.html
https://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/18/lecture-18.pdf
https://www.stat.cmu.edu/~ryantibs/advmethods/notes/bootstrap.pdf - Section 1.3

Bootstrapping

Compute the SE at each point

```python
n_bootstrap = 100

nox_values = np.linspace(np.min(X['NOX']), np.max(X['NOX']))

expected_value_bootstrap_replications = []

for _ in range(n_bootstrap):
    X_boot, y_boot = resample(X, y)
    rf_model_boot = RandomForestRegressor(n_estimators=100).fit(X_boot, y_boot)
    
    bootstrap_model_predictions = []
    for n in nox_values:
        X_pdp = X_boot.copy()
        X_pdp['NOX'] = n
        bootstrap_model_predictions.append(np.mean(rf_model.predict(X_pdp)))
    expected_value_bootstrap_replications.append(bootstrap_model_predictions)
    
expected_value_bootstrap_replications = np.array(expected_value_bootstrap_replications)
for ev in expected_value_bootstrap_replications:
    plt.plot(nox_values, ev, color='blue', alpha=.1)

prediction_se = np.std(expected_value_bootstrap_replications, axis=0)

plt.plot(nox_values, pdp_values, label='Model predictions')
plt.fill_between(nox_values, pdp_values - 3*prediction_se, pdp_values + 3*prediction_se, alpha=.5, label='Bootstrap CI')
plt.legend()
plt.ylabel('Median house price')
plt.xlabel('NOX')
plt.title('Partial dependence plot for NOX vs Price for Random Forest')
plt.show()
```

![partial dependence plot with bootstrap error bars](https://raw.githubusercontent.com/lmc2179/lmc2179.github.io/master/assets/img/pdp/5.png)

# When does the PDP represent a causal relationship?

This section uses a bit of language from CI

Note the assumptions from the paper

- NOX is not a cause of any other predictor variables
- The other predictor variables block all back-door paths between NOX and house price

Include all the causal diagrams

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
