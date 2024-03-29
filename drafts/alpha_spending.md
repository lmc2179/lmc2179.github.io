---
layout: post
title: "How to stop an experiment early with alpha spending"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: alpha_spending.png
---

Let me tell you a story - perhaps a familiar one.

> **Product Manager**: Hey `$data_analyst`, I looked at your dashboard! We only kicked off `$AB_test_name` a few days ago, the results look amazing! It looks like the result is already statistically significant, even though we were going to run it for another week.
>
>**Data Analyst**: Absolutely, they're very promising!
>
>**Product Manager**: Well, that settles it, we can turn off the test, it looks like a winner.
>
>**Data Analyst**: Woah, hold on now - we can't do that! 
>
>**Product Manager**: But...why not? Your own dashboard says it's statistically significant! Isn't that what it's for?
>
>**Data Analyst**: Yes, but we said we would collect two weeks of data when we designed the experiment, and the analysis is only valid if we do that. I have to respect the arcane mystic powers of ✨`S T A T I S T I C S`✨!!! 

_**Has this ever happened to you?**_

This is a frustrating conversation for all involved. The PM is trying to play by the rules by looking at the significance of the test

# The issue with stopping early

```python
import numpy as np
from scipy.stats import ttest_ind, norm
import pandas as pd

days_in_test = 14
samples_per_day = 10


def simulate_one_experiment():
    treated_samples, control_samples = np.array([]), np.array([])
    
    simulation_results = []
    
    for day in range(days_in_test):
        treated_samples = np.append(treated_samples, np.random.normal(0, 1, samples_per_day))
        control_samples = np.append(control_samples, np.random.normal(0, 1, samples_per_day))
        result = ttest_ind(treated_samples, control_samples)
        simulation_results.append([day, len(treated_samples), result.statistic, result.pvalue])
        
    simulation_results = pd.DataFrame(simulation_results, columns=['day', 'n', 't', 'p'])
    return simulation_results

from matplotlib import pyplot as plt
import seaborn as sns

plt.plot(simulate_one_experiment().n, simulate_one_experiment().p)
plt.axhline(.05)
plt.show()

n_simulations = 100
false_positives = 0
early_stop_false_positives = 0

for i in range(n_simulations):
    result = simulate_one_experiment()
    if np.any(result['p'] <= .05):
        early_stop_false_positives += 1
    if result.iloc[-1]['p'] <= .05:
        false_positives += 1

print('False positives with full sample:', false_positives / n_simulations)
print('False positives if early stopping is allowed:', early_stop_false_positives / n_simulations)
```

# Some quick fixes

bonferroni - but it decreases the power! Intuition: Correct for multiple comparisons

linear spending - Intuition: Be more skeptical at the beginning, and normally skeptical at the end

# Solving the problem with the OBF Alpha spending function

https://eclass.uoa.gr/modules/document/file.php/MATH301/PracticalSession3/LanDeMets.pdf

```
def obf_alpha(t_prop): return 2 - 2*(norm.cdf(1.96/np.sqrt(t_prop)))
```

# A note on coverage rates
