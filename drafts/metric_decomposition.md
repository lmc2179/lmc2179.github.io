---
layout: post
title: "What customer group drove the change in my favorite metric? Simple but useful decompositions of change over time"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: jellybeans.png
---

_As analytics professionals, we frequently summarize the state of the business with metrics that measure some aspect of its performance. We check these metrics every day, week, or month, and try to understand what changed them. Often we inspect a few familiar subgroups (maybe your customer regions, or demographics) to understand how much each group contributed to the change. This pattern is so common and so useful that it's worth noting some general-purpose decompositions that we can use when we come across this problem. This initial perspective can give us the intuition to plan a deeper statistical or causal analysis._

# Are my sales growing? Which customers are driving it?

A [metric or KPI](https://en.wikipedia.org/wiki/Performance_indicator) is a single-number snapshot of the business that summarizes something we care about. Data Scientists design and track metrics regularly in order to understand how the business is doing - if it's achieving its goals, where it needs to allocate more resources, and whether anything surprising is happen. When these metrics move (whether that move is positive or negative), we usually want to understand _why_ that happened, so we than think about what (if anything) needs to be done about it. A common tactic for doing this is to think about the different segments that make up your base of customers.

A prototypical example is something like a retail store, whose operators make money by selling things to their customers. In order to take a practical look at how metrics might inform our understanding of the business situation, we'll look at [data from UK-based online retailer](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II). As an online retailer, you produce value by selling stuff; you can measure the total volume of stuff you sold by looking at total revenue, and your efficiency by looking at the revenue produced per customer. This kind of retailer might make marketing, product, sales or inventory decisions at the country level, so it would be useful to understand how each country contributed to your sales growth and value growth.

# Where did my revenue come from

$R_t = \sum\limits_g r_t^g$

$\Delta R_t = \sum\limits_g (r_t^g - r_{t-1}^g)$

```
total_rev_df = monthly_df.groupby('date').sum()

plt.plot(total_rev_df.index, total_rev_df['revenue'] / 1e6, marker='o')
plt.title('Monthly revenue')
plt.xlabel('Month')
plt.ylabel('Total Revenue, millions')
plt.show()

plt.plot(total_rev_df.index[1:], np.diff(total_rev_df['revenue'] / 1e6), marker='o')
plt.title('Monthly revenue change')
plt.xlabel('Month')
plt.ylabel('Month-over-month revenue change, millions')
plt.axhline(0, linestyle='dotted')
plt.show()
```

```
def decompose_total_rev(df):
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

```
ALL_COUNTRIES = ['United Kingdom', 'Germany', 'France', 'Australia', 'All others']

total_revenue_factors_df = decompose_total_rev(monthly_df)

for c in ALL_COUNTRIES:
  plt.plot(total_revenue_factors_df['date'], total_revenue_factors_df[c], label=c)
plt.legend()
plt.show()

for c in ALL_COUNTRIES:
  plt.plot(total_revenue_factors_df['date'].iloc[1:], 
           total_revenue_factors_df[c].iloc[1:] / np.diff(total_revenue_factors_df['total']), 
           label=c)
plt.legend()
plt.show()

# largest driver country list
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
value_per_customer_series = monthly_df.groupby('date').sum()['revenue'] / monthly_df.groupby('date').sum()['n_customers']
plt.plot(value_per_customer_series.index, value_per_customer_series)
plt.show()

value_per_customer_series = monthly_df.groupby('date').sum()['revenue'] / monthly_df.groupby('date').sum()['n_customers']
plt.plot(value_per_customer_series.index[1:], np.diff(value_per_customer_series), marker='o')
plt.axhline(0, linestyle='dotted')
plt.show()
```

```python
def decompose_value_per_customer(df):
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

```
customer_value_breakdown_df = decompose_value_per_customer(monthly_df)

plt.plot(customer_value_breakdown_df.dates.iloc[1:], 
         customer_value_breakdown_df['a'].iloc[1:], marker='o')
plt.plot(customer_value_breakdown_df.dates.iloc[1:], 
         customer_value_breakdown_df['b'].iloc[1:], marker='o')
plt.axhline(0, linestyle='dotted')
plt.show() # Mostly within-country value changes, rather than mix

plt.plot(customer_value_breakdown_df.dates.iloc[1:], 
         customer_value_breakdown_df['a'].iloc[1:] / np.diff(customer_value_breakdown_df['value']), marker='o')
plt.plot(customer_value_breakdown_df.dates.iloc[1:], 
         customer_value_breakdown_df['b'].iloc[1:] / np.diff(customer_value_breakdown_df['value']), marker='o')
plt.axhline(0, linestyle='dotted')
plt.show()

for c in ALL_COUNTRIES:
  plt.plot(customer_value_breakdown_df['dates'].iloc[1:], 
           (customer_value_breakdown_df[c+'_a'].iloc[1:] + customer_value_breakdown_df[c+'_b'].iloc[1:]) / np.diff(customer_value_breakdown_df['value']), 
           label=c)
plt.legend()
plt.show()

# Australia contributed disproportionately positively in August, because Australians became more valuable customers in August
# Correlations between country contributions?
```

# This decomposition does not tell us about causal relationships

?

# Quantifying uncertainty

?

https://www.casact.org/pubs/forum/00wforum/00wf305.pdf

# Appendix: Import statements and data cleaning

```
curl https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx --output online_retail.xlsx
```

```
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

retail_df = pd.read_excel('online_retail.xlsx')

begin = pd.to_datetime('2011-01-01 00:00:00', yearfirst=True)
end = pd.to_datetime('2011-12-01 00:00:00', yearfirst=True)

retail_df = retail_df[(retail_df['InvoiceDate'] > begin) & (retail_df['InvoiceDate'] < end)]

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
