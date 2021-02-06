---
layout: post
title: "What customer group drove the change in my favorite metric? Simple but useful decompositions of change over time"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: jellybeans.png
---

_As analytics professionals, we frequently summarize the state of the business with metrics that measure some aspect of its performance. We check these metrics every day, week, or month, and try to understand what changed them. Often we inspect a few familiar subgroups (maybe your customer regions, or demographics) to understand how much each group contributed to the change. This pattern is so common and so useful that it's worth noting some general-purpose decompositions that we can use when we come across this problem. This initial perspective can give us the intuition to plan a deeper statistical or causal analysis._

# What's happening to my sales

https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

```
curl https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx --output online_retail.xlsx
```

```
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

retail_df = pd.read_excel('online_retail.xlsx')

COUNTRIES = {'United Kingdom', 'France', 'Australia', 'Germany'}

retail_df['country_coarse'] = retail_df['Country'].apply(lambda x: x if x in COUNTRIES else 'All others')
retail_df['date'] = retail_df['InvoiceDate'].apply(lambda x: x.month)
retail_df['revenue'] = retail_df['Quantity'] * retail_df['UnitPrice']
# Add number of customers in this country

monthly_gb = retail_df[['date', 'country_coarse', 'revenue', 'CustomerID']].groupby(['date', 'country_coarse'])
monthly_df  = pd.DataFrame(monthly_gb['revenue'].sum())
monthly_df['n_customers'] = monthly_gb['CustomerID'].nunique()
monthly_df = monthly_df.reset_index()
```

# Where did my revenue come from

$R_t = \sum\limits_g r_t^g$

$\Delta R_t = \sum\limits_g (r_t^g - r_{t-1}^g)$

```
def total_rev(df):
  dates, date_dfs = zip(*[(t, t_df.sort_values('country_coarse').reset_index()) for t, t_df in df.groupby('date', sort=True)])
  first = date_dfs[0]
  groups = first['country_coarse']
  columns = ['total'] + list(groups)
  result_rows = np.empty((len(dates), len(groups)+1))
  result_rows[0][0] = first['revenue'].sum()
  result_rows[0][1:] = np.nan
  for t in range(1, len(result_rows)):
    result_rows[t][0] = date_dfs[t]['revenue'].sum()
    result_rows[t][1:] = date_dfs[t]['revenue'] - date_dfs[t-1]['revenue']
  result_df = pd.DataFrame(result_rows, columns=columns)
  result_df['date'] = dates
  return result_df
```

# Why did my value per customer change

$V_t = \frac{\sum\limits_g r^g_t}{\sum\limits_g c^g_t}$

$V^g_t = \frac{r^g_t}{c^g_t}$

$\Delta V_t = \alpha_t + \beta_t = \sum\limits_g (\alpha^g + \beta^g)$

$c_t = \sum\limits_g c_t^g$

$r_t = \sum\limits_g r_t^g$

$\alpha^g = V_{t-1}^g (\frac{c_t^g}{c_t} - \frac{c_{t-1}^g}{c_{t-1}})$

Apply new mix without changing the value per segment

$\beta^g = (V_t^g - V_{t-1}^g) (\frac{c_t^g}{c_t})$

Apply new segment values without changing mix

```python
def value_per_customer(df):
  dates, date_dfs = zip(*[(t, t_df.sort_values('country_coarse').reset_index()) for t, t_df in df.groupby('date', sort=True)])
  first = date_dfs[0]
  groups = first['country_coarse']
  columns = ['value', 'a', 'b'] + ['{0}_a'.format(g) for g in groups] + ['{0}_b'.format(g) for g in groups]
  result_rows = np.empty((len(dates), len(columns)))
  cust_t = pd.Series([dt_df['n_customers'].sum() for dt_df in date_dfs])
  rev_t = pd.Series([dt_df['revenue'].sum() for dt_df in date_dfs])
  value_t = rev_t / cust_t
  result_rows[:,0] = value_t
  result_rows[0][1:] = np.nan
  for t in range(1, len(result_rows)):
    cust_t_g = date_dfs[t]['n_customers']
    rev_t_g = date_dfs[t]['revenue']
    value_t_g  = rev_t_g / cust_t_g
    cust_t_previous_g = date_dfs[t-1]['n_customers']
    rev_t_previous_g = date_dfs[t-1]['revenue']
    value_t_previous_g  = rev_t_previous_g / cust_t_previous_g
    a_t_g = value_t_previous_g * ((cust_t_g / cust_t[t]) - (cust_t_previous_g / cust_t[t-1]))
    b_t_g = (value_t_g - value_t_previous_g) * (cust_t_g / cust_t[t])
    result_rows[t][3:3+len(groups)] = a_t_g
    result_rows[t][3+len(groups):] = b_t_g
    result_rows[t][1] = np.sum(a_t_g)
    result_rows[t][2] = np.sum(b_t_g)
  result_df = pd.DataFrame(result_rows, columns=columns)
  result_df['dates'] = dates
  return result_df
```

# This decomposition does not tell us about causal relationships

?

# Quantifying uncertainty

?

https://www.casact.org/pubs/forum/00wforum/00wf305.pdf
