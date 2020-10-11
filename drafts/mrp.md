```python
import pandas as pd

region_pop_weight = pd.DataFrame({'name': ['A', 'B', 'C', 'D', 'E'], 'weight': [0.4, 0.3, 0.2, 0.05, 0.05], 'key': 0})
frequency_pop_weight = pd.DataFrame({'name': [1, 2, 3, 4, 5], 'weight': [.15, .2, .3, .25, .1], 'key': 0})

all_groups_pop_weight = pd.merge(region_pop_weight, frequency_pop_weight, on='key', suffixes=('_region', '_frequency'))
all_groups_pop_weight['weight'] = all_groups_pop_weight['weight_region'] * all_groups_pop_weight['weight_frequency']

region_pop_weight.drop('key', inplace=True, axis=1)
frequency_pop_weight.drop('key', inplace=True, axis=1)
all_groups_pop_weight.drop('key', inplace=True, axis=1)

region_sample_weight = pd.DataFrame({'name': ['A', 'B', 'C', 'D', 'E'], 'weight': [0.05, 0.4, 0.3, 0.2, 0.05], 'key': 0})
frequency_sample_weight = pd.DataFrame({'name': [1, 2, 3, 4, 5], 'weight': [.1, .15, .2, .25, .3], 'key': 0})

all_groups_sample_weight = pd.merge(region_sample_weight, frequency_sample_weight, on='key', suffixes=('_region', '_frequency'))
all_groups_sample_weight['weight'] = all_groups_sample_weight['weight_region'] * all_groups_sample_weight['weight_frequency']

region_sample_weight.drop('key', inplace=True, axis=1)
frequency_sample_weight.drop('key', inplace=True, axis=1)
all_groups_sample_weight.drop('key', inplace=True, axis=1)

region_approve_rate = pd.DataFrame({'name': ['A', 'B', 'C', 'D', 'E'], 'rate': [.3, .5, .6, .3, .5], 'key': 0})
frequency_approve_rate = pd.DataFrame({'name': [1, 2, 3, 4, 5], 'rate': [.9, .8, .5, .3, .1], 'key': 0})

all_groups_approve_rate = pd.merge(region_approve_rate, frequency_approve_rate, on='key', suffixes=('_region', '_frequency'))
all_groups_approve_rate['rate'] = 0.5*(all_groups_approve_rate['rate_region'] + all_groups_approve_rate['rate_frequency'])

region_approve_rate.drop('key', inplace=True, axis=1)
frequency_approve_rate.drop('key', inplace=True, axis=1)
all_groups_approve_rate.drop('key', inplace=True, axis=1)
```
