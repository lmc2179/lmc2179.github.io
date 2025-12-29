---
layout: post
title: "Autocorrelation and standard error"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: bad_day.png
---

*See the 4x squares notebook for a detailed view and pseudocode of the simulations to be implemented*

# Autocorrelation is everywhere, and if you ignore it your confidence intervals will all be wrong

We're always interested in how metrics are changing over time. Are sales up quarter-over-quarter? Year-on-year? Do we need more coffee beans than last month?

It's tempting to answer these questions with our usual tools, estimating the daily mean and standard error for each period and running a T-Test, or computing confidence intervals.

There is something a little suspect here, though. If we read the fine print, all of our usual calculations of standard errors, confidence intervals, etc are all conditional on the data generating process being "IID". If we look at the daily sales numbers, that doesn't seem very likely - it would be pretty surprising if today, yesterday, and this time last year had the same distribution. And if we were to do time series modeling, a model which made this assumption would do pretty poorly - we know that most real life time series data has some trend, seasonality, etc.

This turns out not to just be a little technical detail. If your data contains autocorrelation, then your standard errors, confidence intervals, and everything else will be off - by a lot!

First we'll talk about why this happens, and then we'll talk about how to fix it.

# Best laid plans: How autocorrelation affects the standard error

Little sapling data scientists, coming up in the Garden of Analysis, learn early on that their best friend is the **sample mean**:

$$\hat{\mu} = \frac{1}{n} \sum_{i=1}^{i=n} y_i$$

They also learn that statements about the sample mean are most useful if you calculate the **standard error of the sample mean**:

$$\hat{SE}(\hat{\mu})= \frac{\sigma}{\sqrt{n}}$$

This calculation is at the heart of many practical applications, especially, clinical trials and A/B tests. It's a powerful tool, letting practitioners make inferences about populations in a very wide variety of situations. Armed with the sample mean, we can compute confidence intervals, hypothesis tests, and lots of other practically useful things.

We often justify this usage of the standard error by referring in some vague way to the Central Limit Theorem, which we are assured applies...most of the time. Specifically, the Central Limit Theorem says something along the lines of

If you collect some data $y_1, ..., y_n$ from an IID process with mean $\mu$ and a variance $\sigma^2$...

...then the distribution of the sample mean $\hat{\mu}$ **converges in distribution** to $N(\mu, \frac{\sigma^2}{n})$ when the sample size $n$ is large.

Practitioners usually gloss over some of the details at this stage (at their own risk, see {footnote}), and then use the usual estimate of $\hat{SE}(\hat{\mu})$ mentioned above. In a lot of practical cases, this works just fine. Here's an example of the sampling distribution of $n=100$ draws from an exponential distribution with true mean $\mu = 1$.

```python
from scipy.stats import expon, sem
import numpy as np

n_simulations = 100
n_samples_per_sim = 100

results = []

covered = 0

for i in range(n_simulations):
	sample = expon(1).rvs(n_samples_per_sim)
	sample_mean = np.mean(sample)
	sample_se = sem(sample)
	if sample_mean - 1.96*sample_se <= 1 <= sample_mean + 1.96*sample_se:
		covered += 1
coverage = covered / n_simulations
print('The coverage is {}'.format(coverage))
```

This confirms what we expect, ie that the usual method works for a garden variety case like an exponential distribution


What if we add mild autocorrelation, like 0.1. This is realistic; what about time series metrics, etc. You definitely have seen this

Oh no our coverage rate suddenly doesn't look the way we want

The issue here is that it's not IID. Each data point contains "less information" than the IID case, since it's correlated with the last one

# Fix 1: Use Robust standard errors

Robust standard errors. Cons: I think they're wider?

# Fix 2: Model Autocorrelation directly

Add lags to the model for as many periods as there is autocorrelation. This makes the \epsilon_t IID again. Con: Requires you to have a pretty good idea of how much autocorrelation there is; but maybe use an correlogram (autocorrelogram?) for that

# Fix 3: Block bootstrap

Demonstrate that the block bootstrap does this correctly

# Appendix: Simulating the sampling distribution of $\hat{\mu}$ for both the IID and autocorrelated process

# Appendix: Tools for assessing autocorrelation - the ACF and PACF

