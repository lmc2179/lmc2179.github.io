
https://data.ny.gov/Transportation/MTA-Subway-Customer-Journey-Focused-Metrics-Beginn/r7qk-6tcy/about_data 

Title: Finding New York's Hottest Train with time series decomposition

# Smoothing time series happens all the time

A shocking number of real world decisions are made by people on a zoom call squinting at a time series chart of a metric and saying "okay, I think I know what's going on here". But do they? In their defense, a time series can be hard to read. For example, the other day as I was sitting on the subway to go to my office (where I would sit on zoom calls squinting at time series plots), I found myself wondering whether there are more people on the subway than there had been a few years ago. Had the subway ridership increased? Well, that's easy, we can grab a data set from our good friends at the MTA, and look at train ridership over time. For example, here's the R train:

{plot of the R train ridership over time}

Hm. I don't know about you, but I do not find this to be an easy chart to read. It looks like it has increased, but how much? Is there like, a dip in the middle there? It's really not obvious.

# proof of concept

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
                    ('4', 'green'), ('5', 'green'), ('6', 'green'), ('G', 'green'),
                    ('7', 'purple'),
                    ('A', 'blue'), ('C', 'blue'), ('E', 'blue'),
                    ('R', 'yellow'), ('N', 'yellow'), ('Q', 'yellow'), ('W', 'yellow'),
                    ('R', 'gray'), ]:
    line_peak_df = df[df['line']==line]
    line_peak_df.index = pd.to_datetime(line_peak_df.month)
    
    from matplotlib import pyplot as plt
    import seaborn as sns
   
    plt.rcParams["figure.figsize"] = (12, 10)
    
    from statsmodels.tsa.seasonal import MSTL, STL
    res = MSTL(line_peak_df['num_passengers'], periods=(12)).fit()
    
    plt.plot(res.trend, label=line, color=color)
    print(line, res.trend.iloc[-1] / res.trend.iloc[0] - 1)

plt.legend()
plt.show()

res.plot()
plt.show()

plt.plot(line_peak_df.num_passengers, marker='o', markersize=12)
plt.plot(line_peak_df.num_passengers.ewm(alpha=0.2).mean())
plt.show()
```