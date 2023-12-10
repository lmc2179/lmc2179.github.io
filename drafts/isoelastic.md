---
layout: post
title: "Elasticity and log-log models for practicing data scientists"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: important_relationship.png
---

_Models of [elasticity](https://en.wikipedia.org/wiki/Elasticity_(economics)) and log-log relationships seem to show up over and over in my work. Since I have only a fuzzy, gin-soaked memory of Econ 101, I always have to remind myself of the properties of these models. The commonly used  $y = \alpha x ^\beta$  version of this model ends up being pretty easy to interpret, and had wide applicabilty across many domains that actual data scientists work._

# It's everywhere!

I have spent a shocking percentage of my career drawing some version of this diagram on a whiteboard:

![image](https://raw.githubusercontent.com/lmc2179/lmc2179.github.io/master/assets/img/important_relationship.png)

This relationship has a few key aspects that I notice over and over again:
* The output increases when more input is added; the line slopes up.
* Each input added is less efficient than the last; the slope is decreasing.
* Inputs and outputs are both positive

If you're an economist, or even if you just took econ 101, you likely recognize . Economists seem to be familiar with a lot of the useful properties of this kind of relationship 

https://en.wikipedia.org/wiki/Isoelastic_function

a common way to formulate this is as $ax^b$

```python
x = np.linspace(.1, 3)

def f(x, a, b):
  return a*x**b

plt.title('Examples of ax^b')
plt.plot(x, f(x, 1, 0.5), label='a=1,b=0.5')
plt.plot(x, f(x, 1, 1.5), label='a=1,b=1.5')
plt.plot(x, f(x, 1, 1.0), label='a=1,b=1.0')
plt.plot(x, f(x, 3, 0.5), label='a=2,b=0.5')
plt.plot(x, f(x, 3, 1.5), label='a=2,b=1.5')
plt.plot(x, f(x, 3, 1.0), label='a=2,b=1.0')
plt.legend()
plt.show()
```
b affects the curvature (when b<1, diminishing returns; when b>1 increasing returns, when b=1 then it's linear), a affects the vertical scale

Nonetheless, I am not an economist (though I've had the pleasure of working with plenty of brilliant people with economics training). If you're like me, then you might not have these details close to hand. This post is meant to be a small primer for anyone who needs to build models with these kinds of functions

We usually want to know this relationship so we can answer some practical questions such as:
* How much input will we need to add in order to reach our desired level of output?
* If we have some free capital, material, or time to spend, what will we get for it? Should we use it here or somewhere else?
* When will it become inefficient to add more input, ie when will the value of the marginal input be less than the marginal output?

Lots and lots and _lots_ of measured variables seem to have this relationship. The notion of elasticity is ubiquitous in economics, which seems to be the first usage of this kind of model that I can find. However, in my own career I've seen this shape of input-output relationship show up over and over:
* Marketing spend and impressions
* Number of users who see something vs the number who engage with it
* Number of samples vs model quality
* Time spent on a project and quality of result
* Size of an investment vs revenue generated (this one was popularized and explored by [a well known early data scientist](https://en.wikipedia.org/wiki/Tendency_of_the_rate_of_profit_to_fall))

A common model for all these things which has the  [Cobb-Douglas production function](https://en.wikipedia.org/wiki/Cobb%E2%80%93Douglas_production_function).

# Some useful facts about the $y = \alpha x ^\beta$ model

## It makes it easy to talk about % change in input vs % change in output

One of the many reasons that the common OLS model $y = \alpha + \beta x$ is so popular is that it lets us make a very succinct statement about the relationship between $x$ and $y$: "A one-unit increase in $x$ is associated with an increase of $\beta$ units of $y$." The usual linear model would be a poor fit for the relationship at hand, so we're using a slightly different one:

$f(x) = y = \alpha x ^ \beta$

The interpretation of this model is a little different

$\frac{f(xm)}{f(x)} = \frac{\alpha (xm)^\beta}{\alpha x ^ \beta} = m^\beta$

> Under this model, multiplying the input by _m_ multiplies the output by $m^\beta$.

Or, if you are percentage afficionado:

> Under this model, changing the input by $p%$ changes the output output by $(1+p%)^\beta$.

## It's easy to fit with OLS because $log y = log \alpha + \beta log x$

Another reason that the OLS model is so popular is because it is easy to compute in practice. This is probably the reason that OLS is so popular at all; the OLS model may not always be true, but it is always possible to compute it.

Compare with usual OLS interpretation (one unit increase --> output change)

## We can use it to solve for input if we know the desired level of output

In practice, we often start with the desired quantity of output, and then try to understand if the required input is available or feasible. It's handy to have a closed form which inverts our model: 

$f^{-1}(y) = (y/\alpha)^{\frac{1}{\beta}}$

If we want to know how a change in the output will require change in the input

$\frac{f^{-1}(ym)}{f^{-1}(y)} = m^{\frac{1}{\beta}}$

?

# An example: Lotsize vs houes price

```python
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd 
from statsmodels.api import formula as smf

df = pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/AER/HousePrices.csv')
df = df.sort_values('lotsize')

#%%
model = smf.ols('np.log(price) ~ np.log(lotsize)', df).fit()

sns.regplot(df['lotsize'], df['price'], fit_reg=False)
plt.plot(df['lotsize'], np.exp(model.fittedvalues))
plt.show()

#%%
sns.regplot(df['lotsize'], df['price'], fit_reg=False)
plt.plot(df['lotsize'], np.exp(model.fittedvalues))
plt.xscale('log')
plt.yscale('log')
plt.show()

#%%

b = model.params['np.log(lotsize)']
a = np.exp(model.params['Intercept'])
print('1% increase in lotsize -->', round(100*(1.01**b-1), 2), '% increase in price')
print('2% increase in lotsize -->', round(100*(1.02**b-1), 2), '% increase in price')
print('10% increase in lotsize -->', round(100*(1.10**b-1), 2), '% increase in price')

#%%

lotsize_pct_change = .3
before_lotsize = 4000
after_lotsize = before_lotsize + before_lotsize*lotsize_pct_change 

before_price_model, after_price_model = np.exp(model.predict(pd.DataFrame({'lotsize': [before_lotsize, after_lotsize]})))
print(before_price_model, after_price_model, after_price_model/before_price_model-1) # The percent change in the model value
print((1+lotsize_pct_change) ** b - 1) # Is the same as we get from the closed form

print(after_lotsize, (after_price_model/a)**(1/b)) # Successful inversion demo

```

Calculate value of 1% increase, compare with cost; or calculate when marginal output < some value

# does this model really describe reality? some causal considerations and notes on modeling

The above analysis is, when you get down to it, mostly algebra. It's useful algebra to be sure, but it is really just repeated manipulation of the functional form $\alpha x ^ \beta$. 

we should not mistake analytical convenience for truth; we need to check both the goodness of fit of the model to the data at hand, and consider the causal assumptions that our counterfactuals rest on

# Some other options

As I mentioned above, the log-log model isn't the only game in town. 

Related concepts
https://en.wikipedia.org/wiki/Arc_elasticity
https://en.wikipedia.org/wiki/Elasticity_of_a_function

Some other options, esp. w/ zeros:

Isotonic regression

IHS
https://marcfbellemare.com/wordpress/wp-content/uploads/2019/02/BellemareWichmanIHSFebruary2019.pdf
https://worthwhile.typepad.com/worthwhile_canadian_initi/2011/07/a-rant-on-inverse-hyperbolic-sine-transformations.html

Maybe sqrt

https://en.wikipedia.org/wiki/Output_elasticity

# Appendix: Estimating when you have only two data points

Occasionally I've gone and computed an elasticity from a single pair of observations

Lets imagine we have only two data points, which we'll call $x_1, y_1, x_2, y_2$. Then, we have two equations and two unknowns, that is:

$$y_1 = \alpha x_1^\beta$$

$$y_2 = \alpha x_2^\beta$$

If we do some algebra, we can come up with estimates for each variable:

$$\beta = \frac{log \ y_1 - log \ y_2}{log \ x_1 - log \ x_2}$$

$$\alpha = exp(log \ y_1 + \beta \ log \ x_1)$$

```python
import numpy as np
def solve(x1, x2, y1, y2):
    # y1 = a*x1**b
    # y2 = a*x2**b
    log_x1, log_x2, log_y1, log_y2 = np.log(x1), np.log(x2), np.log(y1), np.log(y2)
    b = (log_y1 - log_y2) / (log_x1 - log_x2)
    log_a = log_y1 + b*log_x1
    return np.exp(log_a), b
```

Then, we can run an example like this one in which a 1% increase in $x$ leads to a 50% increase in $y$:

```python
a, b = solve(1, 1.01, 1, 1.5)
print(a, b, 1.01**b)
```

Which shows us `a=1.0, b=40.74890715609402, 1.01^b=1.5`.
