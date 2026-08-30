
https://data.ny.gov/Transportation/MTA-Subway-Customer-Journey-Focused-Metrics-Beginn/r7qk-6tcy/about_data 

Extract details for each train. which one has seen major trend changes recently

```python
import pandas as pd

df = pd.read_csv(r'C:\Users\louis\Downloads\MTA_Subway_Customer_Journey-Focused_Metrics__Beginning_2015_20260829.csv')

df = df[df['month'] >= '2023-01-01']
df = df[df['period'] == 'peak']
df['num_passengers'] = df['num_passengers'].str.replace(',', '').astype(float)

for line, color in [('JZ', 'brown'), 
                    ('B', 'orange'), ('D', 'orange'), ('F', 'orange'), ('M', 'orange'), 
                    ('L', 'gray'), 
                    ('1', 'red'), ('2', 'red'), ('3', 'red'), 
                    ('4', 'green'), ('5', 'green'), ('6', 'green'),
                    ('7', 'purple'),
                    ('A', 'blue'), ('C', 'blue'), ('E', 'blue'),
                    ('R', 'yellow'), ('N', 'yellow'), ('Q', 'yellow')]:
    line_peak_df = df[df['line']==line]
    line_peak_df.index = pd.to_datetime(line_peak_df.month)
    
    from matplotlib import pyplot as plt
    import seaborn as sns
   
    plt.rcParams["figure.figsize"] = (12, 10)
    
    from statsmodels.tsa.seasonal import MSTL
    res = MSTL(line_peak_df['num_passengers'], periods=(12)).fit()
    
    plt.plot(res.trend, label=line, color=color)
    print(line, res.trend.iloc[-1] / res.trend.iloc[0] - 1)

plt.legend()
plt.show()
```