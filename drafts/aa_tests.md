---
layout: post
title: "You should run an A/A test"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: calibration.png
---

# A/A testing is an easy win that helps align everyone

Experimentation is a part of just about any Data Science team I've been a part

A/A tests are useful to make sure you're experimenting correctly

Common objection: Running an A/A test is expensive! it's not nearly as expensive as running invalid experiments without realizing it. and IMO you should have a strong prior that something weird will happen because I've never seen it go smoothly the first time. you can't afford not to imo

What does a test include, exactly?

* An assignment (or exposure) procedure
* Confounding variables you want to randomize away
* At least one metric you're going to measure

So, why do it? You can use an A/A test to...

# Demonstrate the effectiveness of your treatment assignment system

Point of assignment check

SRM check - Binomial distribution comparison - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.monte_carlo_test.html with `scipy.stats.binom(n, expected_rate)`

Check for covariate imbalance between treatment and control - Fit propensity model

Check for assigned vs unassigned users. is it actually x%? does unassigned look any different than assigned? are assignments unique?

Measurement lines up across multiple sources

# Confirm the assumptions of your statistical analysis method

Are samples arriving on time

H0

Precision calibration
* Seychelles diagram of control metric vs treatment metric), if paired
* Sample some number of units, graph variance decrease over time
* Does the variance line up with your power analysis

Any SUTVA assumptions

# Create trust with your stakeholders by showing them the results

?

# Fine-tune your tools

?

# Turn those confirmations into automatic checks

?

# Consider an "always on" A/A test

# Appendix: Code to generate the plots

The above plots aren't from real experiments. But in case you're curious how I generated them:

```python
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import norm, binom

plt.xkcd()

n = 1000
p = 0.1
OBSERVED_SAMPLES = 108

simulations = binom(n, p).rvs(10000)


plt.axvline(OBSERVED_SAMPLES, linestyle='dotted')
sns.distplot(simulations, bins=20)
```