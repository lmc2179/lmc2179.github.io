---
layout: post
title: "What customer group drove the change in my favorite metric? Simple but useful decompositions of change over time"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: jellybeans.png
---

_As analytics professionals, we frequently summarize the state of the business with metrics that measure some aspect of its performance. We check these metrics every day, week, or month, and try to understand what changed them. _

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

monthly_df = retail_df[['date', 'country_coarse', 'revenue']].groupby(['date', 'country_coarse']).sum().reset_index()
```

# Where did my revenue come from

?

# Why did my value per customer change

?

# This decomposition does not tell us about causal relationships

?

# Quantifying uncertainty

?

https://www.casact.org/pubs/forum/00wforum/00wf305.pdf
