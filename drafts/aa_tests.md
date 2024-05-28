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
from scipy.stats import norm, binom, monte_carlo_test
import numpy as np

plt.xkcd()

total_samples = 1000
treatment_rate = 0.1
observed_treated_samples = 108

hypothetical_sampling_distribution = binom(total_samples, treatment_rate).rvs

simulations = hypothetical_sampling_distribution(10000)

plt.axvline(observed_treated_samples, linestyle='dotted')
sns.distplot(simulations, bins=20)

def test_for_srm(total_samples, treatment_rate, observed_treated_samples):
    n_simulations = 1000
    expected_treated_samples = treatment_rate * total_samples
    difference_from_expected = np.abs(observed_treated_samples - expected_treated_samples)
    hypothetical_sampling_distribution = binom(total_samples, treatment_rate).rvs
    simulations = hypothetical_sampling_distribution(n_simulations)
    count_at_least_as_extreme = np.sum((simulations <= (expected_treated_samples + difference_from_expected )) & (simulations >= (expected_treated_samples - difference_from_expected ))) 
    p_value = count_at_least_as_extreme / n_simulations
    return p_value

test_result = test_for_srm(total_samples, treatment_rate, observed_treated_samples)
```