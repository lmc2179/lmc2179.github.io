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

```python
plt.scatter(df['temperature'], df['sales'], label='Observed', marker='x', color='black')
plt.xlabel('Temperature (°F)')
plt.ylabel('Sales ($)')
plt.legend()
plt.show()
```

![Scatter plot](https://raw.githubusercontent.com/lmc2179/lmc2179.github.io/master/assets/img/prediction_confidence/1.png)

So far, so good. It looks like the two are correlated positively. In order to understand the relationship a little better, you fit yourself a line using `ols`:

```python
model = smf.ols('sales ~ temperature', df)
results = model.fit()

alpha = .05

predictions = results.get_prediction(df).summary_frame(alpha)
```

And plot it along with the data:

```python
plt.scatter(df['temperature'], df['sales'], label='Observed', marker='x', color='black')
plt.plot(df['temperature'], predictions['mean'], label='Regression line')
plt.xlabel('Temperature (°F)')
plt.ylabel('Sales ($)')
plt.legend()
plt.show()
```


![Regression line](https://raw.githubusercontent.com/lmc2179/lmc2179.github.io/master/assets/img/prediction_confidence/2.png)

The line has a positive slope, as we expect. Of course, this is only a sample of daily temperatures, and we know that there's some uncertainty around the particular regression line we estimated. We visualize this uncertainty by plotting the confidence interval around the predictions:

```python
plt.fill_between(df['temperature'], predictions['mean_ci_lower'], predictions['mean_ci_upper'], alpha=.5, label='Confidence interval')
plt.scatter(df['temperature'], df['sales'], label='Observed', marker='x', color='black')
plt.plot(df['temperature'], predictions['mean'], label='Regression line')
plt.xlabel('Temperature (°F)')
plt.ylabel('Sales ($)')
plt.legend()
plt.show()
```


![Regression line + Confidence interval](https://raw.githubusercontent.com/lmc2179/lmc2179.github.io/master/assets/img/prediction_confidence/3.png)

This tells us something about the uncertainty around the regression line. You'll notice that plenty of `X`s fall outside of the confidence interval. In order to visualize the region where we expect most actual sales to occur, we plot the prediction interval as well:

```python
plt.fill_between(df['temperature'], predictions['obs_ci_lower'], predictions['obs_ci_upper'], alpha=.1, label='Prediction interval')
plt.fill_between(df['temperature'], predictions['mean_ci_lower'], predictions['mean_ci_upper'], alpha=.5, label='Confidence interval')
plt.scatter(df['temperature'], df['sales'], label='Observed', marker='x', color='black')
plt.plot(df['temperature'], predictions['mean'], label='Regression line')
plt.xlabel('Temperature (°F)')
plt.ylabel('Sales ($)')
plt.legend()
plt.show()
```


![Regression line + Confidence interval + Prediction interval](https://raw.githubusercontent.com/lmc2179/lmc2179.github.io/master/assets/img/prediction_confidence/4.png)

We've got quite a dense plot now - let's take some time and walk through all the elements we've added before we tackle them in detail:

- The observed datapoints are `(temperature, sales revenue)` pairs.
- The regression line tells us what the average revenue is as the temperature varies in our dataset. Here, we've assumed that the revenue varies linearly with the temperature. The regression line answers the question: "If we know the temperature, what is our single best guess about the average level of sales we expect to see?"
- The confidence interval tells us the range of the average revenue at a given temperature. It answers the question: "If we know the temperature, what is our uncertainty around the average level of sales?"
- The prediction interval tells us the range where the oberved revenue on an actual day is likely to fall at a given temperature. It answers the question: "If we know the temperature, what actual range of sales might we see on a given day?".

# Recap: What is the OLS model doing?

We can't get into this topic without a bit of non-cat related terminology, so this section will cover that.

In this example, we had two variables: $temperature$ and $sales$. We want to know how $sales$ changes as $temperature$ varies. By convention, we often refer to the outcome we care about as $y$, and explanatory variables like $temperature$ as $x$, or perhaps $X$ if it's a vector. We'll assume it's a vector going forward, because not much changes if we do.

We assume that we can write $y$ as a function of $X$, something like 

$$ y = f(X) + \epsilon $$

The idea is that $f(X)$ tells us how the average of $y$ changes with $X$, and $\epsilon$ is a normally-distributed noise term from all the variables we didn't choose to put in our model. If we knew the exact form of $f(X)$, we could always compute the expected value of $y$ given $X$, namely $\mathbb{E}[y \mid X] = f(X)$.

There are many choices of $f$ we could pick, but for convenience we often assume that $y$ is a linear function of $X$. In that case, we can rewrite the above expression as 

$$ y = \alpha + X \beta + \epsilon$$

Where we've added a vector of parameters called $\beta$ and a scalar called $\alpha$. These parameters are not known to us, but we're going to try and estimate them from the data. There's also another parameter, hidden in this formula - the noise term $\epsilon$ has some standard deviation, which we'll call $\sigma$.

Running the OLS procedure, through the `fit()` function in `statsmodels` or your favorite library, computes estimates of $\alpha$, $\beta$ and $\sigma$. It also computes standard errors and p-values for those parameters. The values that we estimated from the data have that funny little hat symbol - we'll refer to the estimates as $\hat{\alpha}$, $\hat{\beta}$ and $\hat{\sigma}$.

After we've estimated the model, we can compute the "average" value of $y$ given our knowledge of $X$ by computing $\hat{\alpha} + X \hat{\beta}$. So if we have a particular $X_i$ in mind, and we have not yet observed $y_i$, we can predict its expected value. We'll call the predicted value $\hat{y_i}$, to indicate it was derived from the estimated parameter values. So for our linear model, $\hat{y_i} = \hat{\alpha} + X_i \hat{\beta}$.

# Confidence intervals around the predicted mean

## What does the CI for the regression line represent intuitively?

The above explanation walked through the big idea of the OLS process - we estimate $\alpha$ and $\beta$ from the data to get the regression line. These estimates are our "best guesses" at these values, which we sometimes call "point estimates". As is often the case in statistics, there is some uncertainty around those estimates. This uncertainty is represented in classical statistics by confidence intervals, which are derived from the sampling distribution and standard errors. For Bayesians, the story is pretty similar - the uncertainty is represented by credible intervals, which summarize the posterior distribution.

In either case, we're acknowledging that the point estimates $\hat{\alpha}$, $\hat{\beta}$ and $\hat{\sigma}$ leave a lot of information out. Specifically, the point estimates alone don't tell us about how precise we think our estimates are. In addition to the point esimates we have standard errors for each one, and could compute a confidence interval for each. This uncertainty about $\hat{\alpha}$, $\hat{\beta}$ and $\hat{\sigma}$ translates into some uncertainty about the predicted value $\hat{y}$. Let's look once more at the graph from our example:

![Regression line + Confidence interval + Prediction interval](https://raw.githubusercontent.com/lmc2179/lmc2179.github.io/master/assets/img/prediction_confidence/4.png)

The blue line is the regression line, our point estimate - our single best guess about the average relationship between X and y. However, there are a range of regression lines that seem plausible given the data, even if they are not the absolute best fit. This range of plausible regression lines appears as the CI on the chart, the orange band around the blue regression line. Note that the CI is wider around the edges of the data set, and narrower in the middle. That is, we're less sure about the value of $\hat{y}$ near the edges of the data set. In the notation we used before, when we said $y = f(X) + \epsilon$, the CI indicates that there are a number of different $\hat{f}(X)$ that fit the data.

To go back to our example, if we know the temperature is 40°F, we can see that the CI for the average sales is about 21.00-22.50 USD.

## Where does it come from? The standard error of the predicted mean

We usually don't need to calculate CIs by hand. In the Python code above, we used the `get_prediction` function to get a prediction summary from which we could extract the CIs. However, it's worth understanding how this calculation happens under the hood. By looking at the form of the standard errors for the mean of $\hat{y}$, we can learn a little about what affects the size of the confidence intervals. We'll walk through the case where we have a single independent variable and a single dependent one, since that's easier to talk about. But a lot of this intuition will carry over to the multivariate case.

I'm going to focus on the SE formula and it's implications, rather than its derivation. A much better exposition of the derivation than I could ever give can be found in section 8.1 of [Cosma Shalizi's The Truth About Linear Regression](http://www.stat.cmu.edu/~cshalizi/TALR/TALR.pdf), which is a great resource for everything you might want to know about the details of classical linear models. We'll skip to the good part, the standard errors for the regression line:

$$\hat{SE}(\hat{f}(x)) = \frac{\hat{\sigma^2}}{\sqrt{n-2}} \sqrt{1 + \frac{(x - \bar{x})^2}{S^2_x}}$$

The above contains a lot of technical details, but the gist is that the SE will be smaller when:

- The value we're predicting on ($x$) is near the sample mean ($\bar{x}$)
- The variance ($\sigma^2$) of the noise ($\epsilon$) is small
- The sample size ($n$) is large
- The variance of $x$ ($S^2_x$) is large

The first of these explains why the CI gets larger as we get farther from the middle of the dataset, and explains how different datasets would affect the regression line.

# Prediction intervals

## What does the prediction interval represent intuitively?

If we have some value $X$, then under our model the average value of $y$ is given by $\hat{f}(X)$. However, an actual observation of $y$ might be pretty far from $\hat{f}(X)$, even if our model is correct and the true relationship is linear. This happens all the time, because an individual observation from a distribution can be far from the mean of the distribution. The prediction interval represents the range of actual observed values of $y$ that might show up for any particular $X$.

To go back to our example, if we know the temperature is 40°F, we think on average the sales level is about 22.00 USD. However, on a specific 40°F day, we probably won't see 22.00 USD in sales. Instead, we'll see a range around that value, about 18.00-27.00 USD. This range is called the prediction interval, because it's the interval in which we predict that actual observations will lie.

## Where does it come from?

The prediction interval's variance is given by

[the previous reference](http://www.stat.cmu.edu/~cshalizi/TALR/TALR.pdf)

$S^2_{pred}(x) = \hat{\sigma}^2 \frac{n}{n-2} \left( 1 + \frac{1}{n} + \frac{(x - \bar{x})^2}{n s^2_x} \right)$

Note again that the prediction interval will be wider for data points far from the sample mean

## We can check to see if the prediction intervals have the expected coverage

Check probability of covarage

```python
pi_covers_observed = (predictions['obs_ci_upper'] > df['sales']) & (predictions['obs_ci_lower'] < df['sales'])

print('Coverage proportion: {0}'.format(np.mean(pi_covers_observed)))
```

When we run this, we find that the 95% prediction interval covers the observed value 94% of the time, which is about right where we want it.

We can

```python
sns.regplot(df['temperature'], pi_covers_observed, x_bins=3, logistic=True)
plt.title('Checking the coverage probability vs the temperature')
plt.xlabel('Temperature')
plt.ylabel('Coverage probability')
plt.axhline(0.95, linestyle='dotted', label='Expected coverage')
plt.legend()
plt.show()
```

![Checking coverage vs temperature](https://raw.githubusercontent.com/lmc2179/lmc2179.github.io/master/assets/img/prediction_confidence/5.jpg)

# What about a different GLM, like a logit model?

Logit: https://stackoverflow.com/questions/47414842/confidence-interval-of-probability-prediction-from-logistic-regression-statsmode

Brief mention of delta method, link to p. 69 of http://www.stat.cmu.edu/~cshalizi/TALR/TALR.pdf, see https://en.wikipedia.org/wiki/Delta_method#Multivariate_delta_method
Mention the bootstrap for non-asymptotic stuff

Works out of the box with a GLM for logit models


## Can a logit model have a prediction interval?

Yes but it's not useful

# Bonus: We can also simulate fake data points from our model

```python
pr = results.get_prediction(df)
y_sim = np.random.normal(pr.predicted_mean, pr.se_obs)
```

For logit, sample from distribution of $E[y \mid x]$ and then use np.random.binom

# Appendix: Imports

```python
from scipy.stats import norm
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels.api import formula as smf
import pandas as pd
import numpy as np

n = 100
a = 20
b = 9
s = 2

df = pd.DataFrame({'temperature': np.linspace(0, 110, n), 'sales':a + b*np.linspace(0, 1, n) + norm(0, s).rvs(n)})
```
