---
layout: post
title: "Estimating survey results with poststratification"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: jellybeans.png
---

# The problem: Our sample doesn't look like the population we want to understand

```python
naive_estimate = all_subgroups_df['total_approve'].sum() / all_subgroups_df['total_responders'].sum()
```

# The solution: Post-stratification

# The first step is admitting that you have a problem: Understanding if a sample is non-representative

```python
n = all_subgroups_df['total_responders'].sum()

p_values = [binom_test(r, n, p) for r, p in all_subgroups_df[['total_responders', 'pop_weight']].values]

print(p_values)

binom_cis = [proportion_confint(r, n, method='beta') for r, p in all_subgroups_df[['total_responders', 'pop_weight']].values]
low, high = zip(*binom_cis)
plt.vlines(all_subgroups_df['pop_weight'], low, high)
plt.plot([min(low), max(high)], [min(low), max(high)], color='grey', linestyle='dotted')
```

# Post-stratification with a Logit model in statsmodels

```python
positive_examples_df = all_subgroups_df[['name_region', 'name_frequency']]
positive_examples_df['y'] = 1
positive_examples_df['n'] = all_subgroups_df['total_approve']
negative_examples_df = all_subgroups_df[['name_region', 'name_frequency']]
negative_examples_df['y'] = 0
negative_examples_df['n'] = all_subgroups_df['total_responders'] - all_subgroups_df['total_approve']

logit_df = pd.concat((positive_examples_df, negative_examples_df))

logit_model = smf.glm('y ~ name_region + C(name_frequency)', logit_df, n_trials=logit_df['n'], family=sm.families.Binomial())
logit_fit = logit_model.fit()

logit_predictions = logit_fit.get_prediction(all_subgroups_df).summary_frame()

print('Point estimate: ', np.dot(all_subgroups_df['pop_weight'], logit_predictions['mean']))
print('Point SE: ', np.sqrt(np.dot(all_subgroups_df['pop_weight']**2, logit_predictions['mean_se']**2)))
```

# What assumptions did we make just now?

## We have controlled for all the covariates

And this is not testable

## The true model is the logit model

This is not ideal, though it's not ridiculous

# Preparing for a multilevel model - a Bayesian Logit model with PyMC3

```python
unique_regions = all_subgroups_df['name_region'].unique()
region_lookup = {v: i for i, v in enumerate(unique_regions)}
region_idx = [region_lookup[v] for v in all_subgroups_df['name_region']]

unique_freq = all_subgroups_df['name_frequency'].unique()
freq_lookup = {v: i for i, v in enumerate(unique_freq)}
freq_idx = [freq_lookup[v] for v in all_subgroups_df['name_frequency']]

with pm.Model() as unpooled_model:
  a_region = pm.Normal('a_region', 0, sigma=100, shape=len(unique_regions))
  a_freq = pm.Normal('a_freq', 0, sigma=100, shape=len(unique_freq))
  
  response_est = a_region[region_idx] + a_freq[freq_idx]
  x = pm.Binomial('x', n=all_subgroups_df['total_responders'], p=pm.math.invlogit(response_est), observed=all_subgroups_df['total_approve'])
  unpooled_trace = pm.sample(2000)

predicted_responses = []

for a_r, a_f in zip(unpooled_trace['a_region'], unpooled_trace['a_freq']):
  predicted_responses.append(expit(a_r[region_idx] + a_f[freq_idx]))
  
predicted_responses = np.array(predicted_responses)

poststratified_outcomes = np.array([np.dot(r, all_subgroups_df['pop_weight']) for r in predicted_responses])

all_subgroups_df['mean_unpooled'] = np.mean(predicted_responses, axis=0)
all_subgroups_df['low_unpooled'] = np.quantile(predicted_responses, .025, axis=0)
all_subgroups_df['high_unpooled'] = np.quantile(predicted_responses, .975, axis=0)
```

```python
plt.scatter(all_subgroups_df['total_approve'] / all_subgroups_df['total_responders'], all_subgroups_df['mean'])
plt.vlines(all_subgroups_df['total_approve'] / all_subgroups_df['total_responders'], all_subgroups_df['low'], all_subgroups_df['high'])
plt.plot([0, 1],[0,1], linestyle='dotted')
plt.show()

```

# Hierarchical logit


# Appendix: Imports and data generation

```python
import pandas as pd
import numpy as np
import pymc3 as pm
from scipy.special import expit
from statsmodels.stats.proportion import proportion_confint

region_df = pd.DataFrame({'name': ['A', 'B', 'C', 'D', 'E'], 
                                  'pop_weight': [0.4, 0.3, 0.2, 0.05, 0.05], 
                                  'sample_weight': [0.05, 0.4, 0.3, 0.2, 0.05],
                                  'approve_rate': [.3, .5, .6, .3, .5],
                                  'key': 0})
frequency_df = pd.DataFrame({'name': [1, 2, 3, 4, 5], 
                                     'pop_weight': [.15, .2, .3, .25, .1], 
                                     'sample_weight': [.1, .15, .2, .25, .3],
                                     'approve_rate': [.9, .8, .5, .3, .1],
                                     'key': 0})

all_subgroups_df = pd.merge(region_df, frequency_df, on='key', suffixes=('_region', '_frequency'))
all_subgroups_df['pop_weight'] = (all_subgroups_df['pop_weight_region'] * all_subgroups_df['pop_weight_frequency'])
all_subgroups_df['sample_weight'] = (all_subgroups_df['sample_weight_region'] * all_subgroups_df['sample_weight_frequency'])
all_subgroups_df['approve_rate'] = 0.5*(all_subgroups_df['approve_rate_region'] + all_subgroups_df['approve_rate_frequency'])

rng = np.random.default_rng(184972)

all_subgroups_df['total_responders'] = rng.multinomial(1000, all_subgroups_df['sample_weight'])
all_subgroups_df['total_approve'] = rng.binomial(all_subgroups_df['total_responders'], all_subgroups_df['approve_rate'])

all_subgroups_df.drop(['key', 'pop_weight_region', 'pop_weight_frequency', 
                              'sample_weight_region', 'sample_weight_frequency', 
                              'approve_rate_region', 'approve_rate_frequency',
                              'sample_weight', 'approve_rate'], inplace=True, axis=1)
```


