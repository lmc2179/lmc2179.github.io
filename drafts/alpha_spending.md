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

there are good reasons to stop early!

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

n_simulations = 100
false_positives = 0
early_stop_false_positives = 0

for i in range(n_simulations):
    result = simulate_one_experiment()
    if np.any(result['p'] <= .05):
        early_stop_false_positives += 1
        color = 'blue'
        alpha = 0.5
    else:
        color = 'grey'
        alpha = .1
    if result.iloc[-1]['p'] <= .05:
        false_positives += 1
    plt.plot(result['n'], result['p'], color=color, alpha=alpha)

plt.axhline(.05)
plt.xlabel('Number of samples')
plt.ylabel('P-value')
plt.title('Many experiments will cross p < 0.05 even when H0 is true')
print('False positives with full sample:', false_positives / n_simulations)
print('False positives if early stopping is allowed:', early_stop_false_positives / n_simulations)
```

```python
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, norm
from tqdm import tqdm

def constant_alpha(desired_final_alpha, proportion_sample_collected):
    return desired_final_alpha

def linear_alpha_spending(desired_final_alpha, proportion_sample_collected):
    return proportion_sample_collected * desired_final_alpha

def obf_alpha_spending(desired_final_alpha, proportion_sample_collected):
    z_alpha_over_2 = norm.ppf(1-desired_final_alpha/2)
    return 2 - 2*(norm.cdf(z_alpha_over_2/np.sqrt(proportion_sample_collected)))

def simulate_one(control_average, alpha_strategies, samples_per_day, number_of_days, desired_final_alpha):
    total_sample_size = samples_per_day * number_of_days
    results = {f.__name__: 0 for f in alpha_strategies}
    control_samples, treatment_samples = np.array([]), np.array([])
    for day in range(number_of_days):
        control_samples = np.concatenate([control_samples, np.random.exponential(scale=control_average, size=samples_per_day)])
        treatment_samples = np.concatenate([treatment_samples, np.random.exponential(scale=control_average, size=samples_per_day)])
        for alpha_strategy in alpha_strategies:
            alpha = alpha_strategy(desired_final_alpha, len(control_samples) / total_sample_size)
            if ttest_ind(control_samples, treatment_samples).pvalue <= alpha:
                results[alpha_strategy.__name__] = 1
    return results

simulation_results = pd.DataFrame([simulate_one(control_average=5, 
                                                alpha_strategies=[constant_alpha, linear_alpha_spending, obf_alpha_spending], 
                                                samples_per_day=100, 
                                                number_of_days=28, 
                                                desired_final_alpha=.05) for _ in tqdm(range(1000))])

print(simulation_results.mean())
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
