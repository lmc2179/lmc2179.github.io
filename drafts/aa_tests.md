---
layout: post
title: "You should run an A/A test"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: calibration.png
---

# A/A testing is an easy win

Experimentation is a part of just about any Data Science team I've been a part

A/A test

Common objection: Running an A/A test is expensive! it's not nearly as expensive as running invalid experiments without realizing it. and IMO you should have a strong prior that something weird will happen because I've never seen it go smoothly the first time. you can't afford not to imo

What does a test include, exactly?

* An assignment (or exposure) procedure
* At least one metric you're going to measure
* Confounding variables you want to randomize away

So, why do it? You can use an A/A test to...

# Demonstrate the effectiveness of your treatment assignment system

SRM

Point of assignment

Check for covariate imbalance between treatment and control

Check for assigned vs unassigned users. is it actually x%? does unassigned look any different than assigned?

Measurement lines up across multiple sources

# Confirm the assumptions of your statistical analysis method

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
from scipy.stats import norm

plt.xkcd()

sns.distplot(norm(0, 1).rvs(10000), bins=10)
```