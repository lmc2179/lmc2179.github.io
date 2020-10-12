```python
import pandas as pd
import numpy as np
import pymc3 as pm
from scipy.special import expit

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

rng = np.random.default_rng()

all_subgroups_df['total_responders'] = rng.multinomial(1000, all_subgroups_df['sample_weight'])
all_subgroups_df['total_approve'] = rng.binomial(all_subgroups_df['total_responders'], all_subgroups_df['approve_rate'])

all_subgroups_df.drop(['key', 'pop_weight_region', 'pop_weight_frequency', 
                              'sample_weight_region', 'sample_weight_frequency', 
                              'approve_rate_region', 'approve_rate_frequency',
                              'sample_weight', 'approve_rate'], inplace=True, axis=1)

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

naive_estimate = all_subgroups_df['total_approve'].sum() / all_subgroups_df['total_responders'].sum()
```
