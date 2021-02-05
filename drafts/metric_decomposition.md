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
# Should I reset the index?
```

# Where did my revenue come from

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

```

```

# This decomposition does not tell us about causal relationships

?

# Quantifying uncertainty

?

https://www.casact.org/pubs/forum/00wforum/00wf305.pdf
