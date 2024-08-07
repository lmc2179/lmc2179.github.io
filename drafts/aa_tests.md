---
layout: post
title: "You should run an A/A test"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: calibration.png
---

# A/A testing is an easy win that helps align everyone

If your organization is collecting data, there's a good chance that you are also doing (or planning on doing) experimentation (or if you prefer, A/B testing). Experimentation is hard! A lot can go wrong, and it requires close coordination between Engineering, Data Science, and Product teams.

Just about every organization I've ever worked for that does A/B tests has also benefitted from an A/A test. A/A tests are useful to make sure you're experimenting correctly, and my view is that if you're doing experimentation, you need to do one. It may even make sense to do it on some regular cadence!

There are a few clear benefits of A/A testing
* Demonstrate the effectiveness of your treatment assignment system
* Confirm the assumptions of your statistical analysis method
* Create trust with your stakeholders

And more besides!

Common objection: Running an A/A test is expensive! it's not nearly as expensive as running invalid experiments without realizing it. and IMO you should have a strong prior that something weird will happen because I've never seen it go smoothly the first time. you can't afford not to imo

What does a test include, exactly?

* **An assignment (or exposure) procedure.** This is the process that randomly assigns treatment status to the units you're experimenting on. You should have a clear idea of under what conditions a unit is assigned, and what percentage of units should be assigned.
* **Confounding variables you want to randomize away.** This is why we do randomization in the first place! The treatment and control groups should look the same. Randomizing should ensure they should have the same mix of user segments, the same average age, all that.
* **At least one metric you're going to measure.** This is the outcome you care about learning about by experimenting. Your revenue, clickthrough rate, throughput, whatever. 

So, why do it? You can use an A/A test to...

# Demonstrate the effectiveness of your treatment assignment system

Point of assignment check

SRM check - Binomial distribution comparison - `scipy.stats.binomtest`

Check for covariate imbalance between treatment and control - Fit propensity model with `smf.logit`

Check for assigned vs unassigned users. is it actually x%? does unassigned look any different than assigned? are assignments unique?

Measurement lines up across multiple sources

# Confirm the assumptions of your statistical analysis method

Are samples arriving on time

Did you make any parametric assumptions, for example about the distribution of the outcome variable? Are they true?

H0

Precision calibration
* Seychelles diagram of control metric vs treatment metric), if paired
* Sample some number of units, graph variance decrease over time
* Does the variance line up with your power analysis

Any SUTVA assumptions

# Create trust with your stakeholders by showing them the results

helps them understand what exactly will happen in your experiments

establishes a baseline

gives a preview of how they like to see result - do your tools support it?

show that you can't reject H0/that you measure a result which is at most ...

# Turn those confirmations into automatic checks

?

# Appendix: Code to generate the plots

The above plots aren't from real experiments. But in case you're curious how I generated them:

```python
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import norm, binom, monte_carlo_test, binomtest
import numpy as np

plt.xkcd()

total_samples = 1000
treatment_rate = 0.1
observed_treated_samples = 108

test_result = binomtest(observed_treated_samples, total_samples, treatment_rate).pvalue

hypothetical_sampling_distribution = binom(total_samples, treatment_rate).rvs

simulations = hypothetical_sampling_distribution(10000)

plt.axvline(observed_treated_samples, linestyle='dotted')
sns.distplot(simulations, bins=20)
plt.title('p={}'.format(round(test_result, 3)))
```