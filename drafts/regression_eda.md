# Forest plots and regression models are great Exploratory Data Analysis tools

# EDA has no clear-cut recipe

# Why do EDA? (What are examples of EDA insights)

- Build intuition about potential intervention points (points of leverage)
- Check assumptions we have about which variables are associated with the outcome
- Challenges: many variables, maybe many subgroups, hard to intuit the "unique impact" of each


# Regression is a great descriptive tool
- Berk: Describing the data is most of what we do
- Coefficients represent partial correlations
- Dummy encoding shows us extreme subgroups

# Isn't this a fishing expedition?
- Isn't everything
- From the Type I (FWER/FDR perspective, we can do some stuff); Data splitting
- Or just live that Bayes life

# Examples: What is associated with high income?

```bash
#https://archive.ics.uci.edu/ml/datasets/Census+Income
curl https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data > census.csv
```

```python
import pandas as pd
from statsmodels.api import formula as smf

df = pd.read_csv('census.csv')
df.columns = 'age workclass fnlwgt education education_num marital_status occupation relationship race sex capital_gain capital_loss hours_per_week native_country high_income'.split(' ')
df['high_income'] = df['high_income'].apply(lambda x: 1 if x == ' >50K' else 0)

model = smf.logit('high_income ~ age + workclass + education + marital_status + age:workclass', df)
results = model.fit()
```

# Why isn't this a causal interpretation? When might it be?

# What else might go into an EDA?

# Appendix: Python functions for forest plots


```python
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def forestplot_sorted(middle, lower, upper, name, colormap):
  df = pd.DataFrame({'mid': middle,
                     'low': lower,
                     'high': upper,
                     'name': name})
  df = df.sort_values('mid')
  df['position'] = -np.arange(len(df))
  colors = colormap(np.linspace(0, 1, len(df)))
  plt.scatter(df['mid'], df['position'], color=colors)
  plt.scatter(df['low'], df['position'], color=colors, marker='|')
  plt.scatter(df['high'], df['position'], color=colors, marker='|')
  plt.hlines(df['position'], df['low'], df['high'], color=colors)
  plt.yticks(df['position'], df['name'])
  
forestplot_sorted([0, 1, 2, 3], [-1, 0, 1, 2], [1, 2, 3, 4], ['a', 'b', 'c', 'd'], plt.cm.plasma)
plt.show()
```


```python
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

def group_covariates(terms, cols):
  groups = -np.ones(len(cols))
  g = 0
  for i, c in enumerate(cols):
    if c[:len(terms[g])] != terms[g]: # Check first part of string
      g +=1
    groups[i] = g
  return groups.astype('int')

def clean_categorical_name(n):
  i = n.index('[')
  return n[i+3:-1]
  
def is_level(group_name, col_name):
  return group_name != col_name

def forestplot(model, fit_results, alpha=.05, cols_to_include=None, bonferroni_correct=False):
  if bonferroni_correct:
    a = alpha / len(fit_results.params)
  else:
    a = alpha
  summary_matrix = pd.DataFrame({'point': fit_results.params,
                                 'low': fit_results.conf_int(a)[0],
                                 'high': fit_results.conf_int(a)[1],
                                 'name': model.data.design_info.column_names,
                                 'position': -np.arange(len(fit_results.params))})
  terms = model.data.design_info.term_names
  n_terms = len(terms)
  term_group = group_covariates(terms, summary_matrix['name'])
  summary_matrix['term'] = [terms[g] for g in term_group]
  term_colors = plt.cm.rainbow(np.linspace(0, 1, n_terms))
  summary_matrix['color'] = [term_colors[g] for g in term_group]
  summary_matrix['clean_name'] = [clean_categorical_name(c) if is_level(t, c) else c for t, c in summary_matrix[['term', 'name']].values]
  if cols_to_include is None:
    cols = set(terms)
  else:
    cols = set(cols_to_include)
  summary_matrix = summary_matrix[summary_matrix['term'].apply(lambda x: x in cols)]
  plt.scatter(summary_matrix['point'], summary_matrix['position'], c=summary_matrix['color'])
  for p, l, h, c in summary_matrix[['position', 'low', 'high', 'color']].values:
    plt.plot([l, h], [p, p], c=c)
  plt.axvline(0, linestyle='dotted', color='black')
  plt.yticks(summary_matrix['position'], summary_matrix['clean_name'])
```

Need: bspline support and interaction support
