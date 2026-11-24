# 

Example: Diamond data set; effect of size (carat) on price

Underlying causal model, questions about how the data was collected

# Coarsen

# Split

# Bin

# Match

# Check match quality

# Estimate average effects and heterogeneous effects

# Estimating the cut vs price curve with CEM

```python
import pandas as pd
import numpy as np

def cem(df, y_col, treatment_status_col, control_level, treated_level, match_cols):
    df_c, df_t = df[df[treatment_status_col] == control_level], df[df[treatment_status_col] == treated_level]
    df_bin_c = df_c.groupby(match_cols).agg(
      average_y=(y_col, 'mean'),
      sd_y=(y_col, 'std'),
      n=(y_col, 'size')
    )
    df_bin_t = df_t.groupby(match_cols).agg(
      average_y=(y_col, 'mean'),
      sd_y=(y_col, 'std'),
      n=(y_col, 'size')
    )
    df_bin_joined = pd.merge(df_bin_c, df_bin_t, on=match_cols, how='outer', suffixes=('_c', '_t'))
    df_bin_matched = df_bin_joined[(df_bin_joined['n_c'] >= 2) & (df_bin_joined['n_t'] >= 2)]
    
    t_match_pct = df_bin_matched['n_t'].sum() / df_bin_joined['n_t'].sum()
    c_match_pct = df_bin_matched['n_c'].sum() / df_bin_joined['n_c'].sum()
    
    sd_c_pooled = np.sqrt((df_bin_matched['sd_y_c']**2 * (df_bin_matched['n_c'] - 1)).sum() / (df_bin_matched['n_c'] - 1).sum())
    sd_t_pooled = np.sqrt((df_bin_matched['sd_y_t']**2 * (df_bin_matched['n_t'] - 1)).sum() / (df_bin_matched['n_t'] - 1).sum())

    df_bin_matched['delta'] = df_bin_matched['average_y_t'] - df_bin_matched['average_y_c']

    weights_att = df_bin_matched['n_t'] / df_bin_matched['n_t'].sum()
    weights_atc = df_bin_matched['n_c'] / df_bin_matched['n_c'].sum()
    weights_ate = (df_bin_matched['n_t']+df_bin_matched['n_c']) / (df_bin_matched['n_t']+df_bin_matched['n_c']).sum()

    delta_att = np.dot(df_bin_matched['delta'], weights_att)
    delta_atc = np.dot(df_bin_matched['delta'], weights_atc)
    delta_ate = np.dot(df_bin_matched['delta'], weights_ate)

    df_bin_matched['standard_error_c'] = sd_c_pooled / np.sqrt(df_bin_matched['n_c'])
    df_bin_matched['standard_error_t'] = sd_t_pooled / np.sqrt(df_bin_matched['n_t'])
    df_bin_matched['standard_error_delta'] = np.sqrt((df_bin_matched['standard_error_c'])**2 + (df_bin_matched['standard_error_t'])**2)

    standard_error_att = np.sqrt(np.dot(weights_att**2, df_bin_matched['standard_error_delta']**2))
    standard_error_atc = np.sqrt(np.dot(weights_atc**2, df_bin_matched['standard_error_delta']**2))
    standard_error_ate = np.sqrt(np.dot(weights_ate**2, df_bin_matched['standard_error_delta']**2))

    treatment_effects  = (delta_att, delta_atc, delta_ate)
    standard_errors = (standard_error_att, standard_error_atc, standard_error_ate)
    match_statistics = (t_match_pct, c_match_pct)
    return treatment_effects, standard_errors, match_statistics

from matplotlib import pyplot as plt

df = pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/ggplot2/diamonds.csv')

df['carat'] = df['carat'].round(1)

y_col = 'price'
X_cols = ['clarity', 'color', 'cut']
t_col = 'carat'

carat_levels = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
results = [cem(df, y_col, t_col, carat_c, carat_t, X_cols) for carat_c, carat_t in zip(carat_levels[:-1], carat_levels[1:])]
att = np.array([r[0][0] for r in results])
se = np.array([r[1][0] for r in results])
plt.plot(carat_levels[1:], att)
plt.vlines(carat_levels[1:], att + 3*se, att - 3*se)
plt.ylim(0)
plt.show()

print(sum(att))

# Carat vs Price
plt.scatter(df['carat'], df['price'])

# All the way from min to max - compare with plots above
cem(df, y_col, t_col, 0.2, 2.0, X_cols)

```

This is still not _really_ the ATE - it is the ATE on the matching subpopulation

# Draft version

```python
import pandas as pd

def coarsened_exact_matching_estimate_of_ate(df, y_col, X_cols, t_col, control_value, treatment_value):
  df_control = df[df[t_col] == control_value].groupby(X_cols).agg(
      average_y=(y_col, 'mean'),
      sd_y=(y_col, 'std'),
      row_count=(y_col, 'size')
  ).reset_index()
  df_treatment = df[df[t_col] == treatment_value].groupby(X_cols).agg(
      average_y=(y_col, 'mean'),
      sd_y=(y_col, 'std'),
      row_count=(y_col, 'size')
  ).reset_index()
  bin_summaries = pd.merge(df_control, df_treatment, on=X_cols, how='outer', suffixes=['_control', '_treatment'])
  # TODO: Remove bins with zero (or maybe stricter), report match diagnostics. Estimate bin effects and average effect. Pool variance and estimate standard errors
  return bin_summaries 
  

# Test data from https://vincentarelbundock.github.io/Rdatasets/doc/ggplot2/diamonds.html
df = pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/ggplot2/diamonds.csv')

df['carat'] = df['carat'].round(1)

y_col = 'price'
X_cols = ['clarity', 'color', 'carat']
t_col = 'cut'

cut_levels = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']

t0_value = 'Premium'
t1_value = 'Ideal'

result = coarsened_exact_matching_estimate_of_ate(df, y_col, X_cols, t_col, t0_value, t1_value)
```
