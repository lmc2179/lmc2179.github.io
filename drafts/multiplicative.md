---
layout: post
title: "When do we log transform the response variable? Model assumptions, multiplicative models and time series decompositions"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: logarithms.png
---

*TL;DR - Sometimes, analysts will perform a log transformation of the outcome variable to "make the residuals look normal". In some cases this is just papering over other issues, but sometimes this kind of transformation genuinely improves the inference or produces a better fitting model. In what cases does this happen? Why does the log transformation work the way it does?*

https://xkcd.com/451/

# One common reason: Because log-transformed data fits the model assumptions better than any other transformation

A commonly cited justification for log transforming the response variable is that the OLS assumptions are being violated, and the transformation will remedy this. These arguments often go something like:

- My residuals are non-normal because they are skewed or have outliers; a log transform makes them more symmetric.
- My residuals show evidence of heteroskedasticity; log transforming the response makes the residual variance seem constant.
- My dependent variable is constrained to be positive, but in an OLS model the outcome variable can be negative.

A log transformation of the response variable may *sometimes* resolve these issues, and is worth considering. However, each of these problems has other potential solutions:

- Asymmetric residuals could be resolved by a different non-linear transformation of the outcome; the log transform is not special. A square root, for example, may have the same effect. You might also consider a model with an asymmetric or heavy-tailed residual distribution.
- Heteroskedasticity can be accounted for by making the non-constant variance part of your model. In the linear model framework, [WLS](http://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/24/lecture-24--25.pdf) is a common solution.
- A dependent variable which is definitionally positive can be accounted for with a GLM other than OLS, like a Negative-binomial model or Gamma model. 

Variance-stabilizing transformations like the [Box-Cox transformation](https://en.wikipedia.org/wiki/Power_transform#Box%E2%80%93Cox_transformation) are also popular methods for dealing with these problems, and are more complex than simply taking a log.

The point here is _not_ that a log transformation can't solve these problems - it sometimes can! Rather, the point is that it will not _always_ solve these problems. It's worth looking at an example where the OLS assumptions are violated but the log transform doesn't help. 



## Log transformations do not automatically fix model assumption problems

It's worth walking through an example where a logarithm transform of the response doesn't work, in order to get some intuition as to why.

Let's look at a dataset on which we apply a simple regression model. The dataset has one binary dependent variable, and one continuous outcome variable. The true distribution of the data is:

$$y | x = 0 ~ N(10, 2)$$

$$y | x = 1 ~ N(20, 1)$$

That is, as $x$ increases, so does the expected value of $y$. And as $x$ increases, the variance of $y$ _decreases_, a classically heteroskedastic situation. Let's construct a sample from this dataset, and then fit a linear model to it. We'll plot the residuals against $x$, a standard diagnostic for heteroskedasticity.

*You can find the import statements for the code at the bottom of the post.

```python
df = pd.DataFrame({'x': [0] * 100 + [1] * 100, 
                   'y': np.random.normal([10] * 100 + [20] * 100, [2]*100 + [1]*100)})

model = smf.ols('y ~ x', df)
fit = model.fit()

plt.scatter(df['x'], df['y'] - fit.predict(df))
plt.show()
```

The result is what we expect - the variance of the residuals changes with $x$. As I mentioned, this is sometimes a case in which a log transform is used. Let's try one out and see why happens with this model's residuals:

```python
log_model = smf.ols('np.log(y) ~ x', df)
log_fit = log_model.fit()

plt.scatter(df['x'], log_fit.resid)
plt.show()

```

Oh my - the problem seems to have gotten _worse_. What happened? Well, the "log transform to create homoskedasticity" trick does sometimes works, but it only works in situations where the variance _increases_ as the predictor increases. In this case, the variance decreases as the predictor increases - a log transform makes the problem _worse_.

Again, this doesn't mean that a log transform of the dependent variable will never solve your problem. This is, instead, a word of caution on doing so - it may not solve your problem, and there are other solutions worth considering.

# Another reason: Because you'd like a model where coefficients combine multiplicatively instead of additively (log-linear model)


An attempt to correct bad OLS assumptions isn't the only reason we might log transform the response variable. Fitting a model like this will change the way that the coefficients combine in the predicted value of y.

Let's consider an example. Imagine that you've recently been spending a lot of time indoors (for some of us this requires very little imagination), time which you've spent making extremely detailed tiny sculptures of cats so you can sell them online. This has proven a surprisingly lucrative revenue stream, and you'd like to do an analysis of the customers on your email list. For each customer, you know a few basic facts:

- How much revenue they produced from their purchases last month, in dollars, which we'll call $y$
- Whether they were a returning customer or a new customer, a binary variable we'll call $X_{returning}$
- Whether they are in country A or country B (you live in country A, but the denizens of country B seem to absolutely adore cat sculptures), another binary variable, $X_{B}$

You'd like to understand the relationship between the revenue of a customer and their demographic variables. A common model for data like this is something like

$$y = \alpha + \beta_{returning} X_{returning} + \beta_{B} X_{B} + \epsilon$$

We can fit this model with OLS, producing $\widehat{\alpha}$, $\widehat{\beta_{returning}}$, and $\widehat{\beta_{B}}$. When we look at this fit, we can inspect these variables and interpret them like this:

- $\widehat{\alpha}$ is the "baseline", the average revenue for new customers in country A.
- $\widehat{\beta_{returning}}$, and $\widehat{\beta_{B}}$ are the expected added revenue from learning that a customer is returning and that a customer is in country B, respectively. We interpret this as the change in the expected value for each variable, "holding all others constant".

This model has three coefficient parameters (we'll avoid talking about the nuisance parameter for noise) and no interaction term. It's totally plausible that there is an interaction between these two dependent variables. Perhaps returning customers produce more revenue than new ones, and country B customers produce more than country A, and a user who is both produces _much_ more revenue. In that case, the interaction term would be positive, indicating that observing both is correlated with more than either attribute counted individually. It's also plausible that the interaction term is negative, indicating that there are "diminishing returns", and the whole is less than the sum of its parts. Either way, adding this kind of interaction to your model will cost you an extra parameter.

Let's now consider an alternative 3 parameter model:

$$log(y) = \alpha + \beta_{returning} X_{returning} + \beta_{B} X_{B} + \epsilon$$

How is this different from the previous model? The parameters now have a different interpretation. For example, $\widehat{\beta_{returning}}$ is no longer the value _added_ to the expected value when we learn a customer is returning. Rather, this now _multiplies_ the expected value by $e^{\widehat{\beta_{returning}}}$. This is sometimes called a [log-linear model](https://en.wikipedia.org/wiki/Log-linear_model#:~:text=A%20log%2Dlinear%20model%20is,(possibly%20multivariate)%20linear%20regression) - the logarithm of y is a linear function of X. 

This kind of model makes sense if:
- Your values of $y$ are strictly positive
- It would be convenient to have a model where you can interpret the coefficients as multipliers, rather than additive changes
- You think that the effects of your covariates combine non-additively. Specifically, they combine multiplicatively in the positive direction.

If this last one is true, your model may fit the data better than the additive model with the same number of parameters.

A comparison of the expected value of each subgroup under each model can be found in the table below.

|Customer type|Country|Expected value under additive model|Expected value under log model|
|---|---|---|---|
|New|A|$\alpha$|$e^{\alpha}$|
|Returning|A|$\alpha + \beta_{returning}$|$e^{\alpha} e^{\beta_{returning}}$|
|New|B|$\alpha + \beta_{B}$|$e^{\alpha} e^{\beta_{B}}$|
|Returning|B|$\alpha + \beta_{returning} + \beta_{B}$|$e^{\alpha} e^{\beta_{returning}} e^{\beta_{B}}$|

The most important entry here is the last one, demonstrating the non-additive combination of regression terms.

# A common use case: Multiplicative time series decomposition

There's a use case for a multiplicative combinations which is common enough that it's worth walking through it here. 

https://en.wikipedia.org/wiki/Decomposition_of_time_series

To see the difference between these two models in action, we're going to look at a [classic time series dataset of monthly airline passenger counts from 1949 to 1960](https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv). Plotting the dataset, we see some common features of time series data: there are clear seasonal trends, and a steady increase year over year.

```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels.api import formula as smf

df = pd.read_csv('airline.csv')

df['year'] = df['Month'].apply(lambda x: int(x.split('-')[0]))
df['month_number'] = df['Month'].apply(lambda x: int(x.split('-')[1]))

additive_model = smf.ols('Passengers ~ year + C(month_number)', df)
additive_fit = additive_model.fit()

multiplicative_model = smf.ols('np.log(Passengers) ~ year + C(month_number)', df)
multiplicative_fit = multiplicative_model.fit()

plt.plot(df['Passengers'])
plt.plot(additive_fit.fittedvalues)
plt.plot(np.exp(multiplicative_fit.fittedvalues))
plt.show()

plt.plot(additive_fit.fittedvalues - df['Passengers'])
plt.plot(np.exp(multiplicative_fit.fittedvalues) - df['Passengers'])
plt.show()
```

# Appendix: Imports

```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels.api import formula as smf
from scipy.stats import norm
```
# Appendix: Further reading

https://stats.stackexchange.com/a/3530/29694

[Cosma Shalizi's excellent regression lecture notes](http://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/07/lecture-07.pdf)
