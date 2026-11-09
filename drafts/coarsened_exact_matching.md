

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
