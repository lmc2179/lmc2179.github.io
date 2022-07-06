Interrupted time series analysis

Possible Example: Did Blazing Saddles kill the western?

Key idea: In the absence of an ironclad causal inference method (experiment, clear list of counfounders, instrumental variable), we often tend to use an informal "before vs after" look to make a guess about causal effects after an intervention is introduced. We might also include our historical knowledge of previous fluctuation levels, pre-treatment trends, and cyclic behavior, and attempt to synthesize them. This article is about ITS, the formal way of doing that.

Nonlinear extensions with B-splines

```python
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
from datetime import datetime
from statsmodels.api import formula as smf
import numpy as np

df = pd.read_csv('https://www1.nyc.gov/assets/tlc/downloads/csv/data_reports_monthly.csv')
df['trips'] = df['         Trips Per Day         '].str.replace(',', '').astype(float)
df['date'] = df['Month/Year'].apply(lambda s: datetime.strptime(s + '-01', '%Y-%m-%d'))

daily_trip_series = df.groupby('date').sum()['trips']

daily_trip_regression_data = pd.DataFrame({'date': daily_trip_series.index, 'trips': daily_trip_series}).reset_index(drop=True).sort_values('date')
#daily_trip_regression_data = daily_trip_regression_data[daily_trip_regression_data['date'] >= '2019-01-01']
#daily_trip_regression_data = daily_trip_regression_data[daily_trip_regression_data['date'] < '2022-01-01']
daily_trip_regression_data['month'] = daily_trip_regression_data['date'].apply(lambda x: x.month)
daily_trip_regression_data['year'] = daily_trip_regression_data['date'].apply(lambda x: x.year)
daily_trip_regression_data['trend'] = np.arange(len(daily_trip_regression_data))
daily_trip_regression_data['after'] = (daily_trip_regression_data['date'] >= '2020-04-01').apply(int)
daily_trip_regression_data['after'].mask(daily_trip_regression_data['date'] == '2020-03-01', 1./3, inplace=True)
daily_trip_regression_data['after_trend'] = np.cumsum(daily_trip_regression_data['after'])

plt.scatter(daily_trip_regression_data['date'], daily_trip_regression_data['trips'])

model = smf.ols('trips ~ trend + after + after_trend', daily_trip_regression_data)
#model = smf.ols('trips ~ bs(trend, df=5) + after + bs(after_trend, df=5)', daily_trip_regression_data)
result = model.fit()
plt.plot(daily_trip_regression_data['date'], result.fittedvalues)

plt.show()
```
