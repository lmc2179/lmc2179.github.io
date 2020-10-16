---
layout: post
title: "Generalizing from biased samples with Multilvel regression with Poststratification using PyMC3"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: jellybeans.png
---

# The problem: Our sample doesn't look like the population we want to understand

In most practical settings, we can't inspect every member of a group of interest. We collect count events, track behaviors, and survey opinions of individuals to make generalizations to the population that the individual came from. This is the process of statistical inference from the data at hand which we spend so much of our time trying to do well. For example, perhaps you run a startup and you'd like to survey your users to understand if they'd be interested in new product feature you've been thinking about. Developing a new feature is pretty costly, so you only want to do it if a part portion of your user base will be interested in it. You send an email survey to a small number of users, and you'll use that to infer what your overall user base thinks of the idea.

In the simplest case, every user has the same likelihood of responding to your survey. In that case, the average member of your sample looks like the average member of your user base, and you have a [simple random sample](https://en.wikipedia.org/wiki/Simple_random_sample) to which you can apply all the usual analyses. For example, in this case the sample approval rate is a reasonable estimate of the population approval rate.

However, simple random sampling is often not an option for someone running such a survey. In most situations like this, every user is not equally likely to respond - for example, there's a good chance that your most enthusiastic users are more likely to respond to the survey. This leaves you with an estimate that over-represents these enthusiastic users. In this case, the usual estimate of the approval rate (the sample approval rate) is not an exact reflection of the population approval rate, because our sampling was not simply random. Instead, we'll need to do some work to uncover how our sample differs from the population of users, so we can account for the bias in our sampling process.

# The solution: Poststratification

We'll correct this problem using a technique called [poststratification](https://online.stat.psu.edu/stat506/lesson/6/6.3). Poststratification goes something like this:

- Collect your sample, which is biased because it oversamples some subgroups and undersamples others. For example, perhaps your survey is likely to be easier to answer for some demographics, but harder for others.
- Use the data to estimate the mean result (such as approval rating) for each subgroup.
- Compute the population mean by calculating a weighted average of the subgroup means, using the proportion of each subgroup in the population as the weight.

The reason that this technique has such a fancy sounding name is because it assumes that users are sample from a bunch of discrete subgroups (the strata), and we are adjusting the observed average after we collected the data using the strata information (so, we are doing the adjustment "after subgroup analysis", or "post-stratification").

?

There are a number of ways that we can perform poststratification. The technique above is about the simplest kind that I can imagine - we estimate the subgroup average as the sample average, and use that for reweighting. However, we can often do a little better than this. In particular, we can get better estimates of the subgroup means by using a Bayesian technique called multilevel (or hierarchical) regresison, leading us to [Multilevel regression with poststratification](https://en.wikipedia.org/wiki/Multilevel_regression_with_poststratification). At this time, MRP is one of the state-of-the-art methods for generalizing samples of public opinion like polls. In 2016, [Wang et al](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/04/forecasting-with-nonrepresentative-polls.pdf) demonstrated the power of this technique by showing that it could be used to accurately predict US national public opinion from a highly nonrepresentative survey. In that case, the sample consisted of responders to an Xbox live poll, which very strongly oversampled certain subgroups (like young men) and undersampled others (like older women). However, using MRP the authors were able to understand the bias in the data and adjust the survey results accordingly.

# What are these subgroups, exactly? Where do they come from?

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
naive_estimate = all_subgroups_df['total_approve'].sum() / all_subgroups_df['total_responders'].sum()
print('The observed approval rate is: {0}'.format(naive_estimate))
```
```
The observed approval rate is: 
```

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


