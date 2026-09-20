---
layout: post
title: "Finding New York's Hottest Train with time series decomposition"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: top_trains.png
---
 

# Lots of very important decisions are made by looking at time series data

A shocking number of real world decisions are made by people on a zoom call squinting at a time series chart of a metric and saying "okay, I think I know what's going on here". But do they? In their defense, a time series can be hard to read. For example, the other day as I was sitting on the subway to go to my office (where I would sit on zoom calls squinting at time series plots), I found myself wondering whether there are more people on the subway than there had been a few years ago. The subway certainly _felt_ more crowded, but maybe I'm just looking through rose tinted glasses at the New York of yesteryear (I wouldn't be the first). Had the subway ridership actually increased? That's a data question! We can grab a [data set](https://data.ny.gov/Transportation/MTA-Subway-Customer-Journey-Focused-Metrics-Beginn/r7qk-6tcy/about_data) from our good friends at the MTA, and look at train ridership over time. For example, here's the **monthly ridership of the L train**, which I was commuting on:

```python
# Import the data I downloaded from https://data.ny.gov/Transportation/MTA-Daily-Ridership-Data-2020-2025/vxuj-8kew/about_data

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from statsmodels.tsa.seasonal import MSTL

plt.rcParams["figure.figsize"] = (10, 8)

df = pd.read_csv(r'MTA_Subway_Customer_Journey-Focused_Metrics__Beginning_2015_20260829.csv')

df = df[df['month'] >= '2023-01-01']
df = df[df['period'] == 'peak']
df['num_passengers'] = df['num_passengers'].str.replace(',', '').astype(float)

# L train plot
def millions(x, pos):
    return f'{x*1e-6:.1f}M'

l_train_peak_df = df[df['line']=='L']
l_train_peak_df.index = pd.to_datetime(l_train_peak_df.month)

plt.title('Monthly Ridership on the (L) train')
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(millions))
plt.plot(l_train_peak_df.num_passengers, marker='o', markersize=12, color='grey', label='Monthly Riders')
plt.legend()
plt.show()
```

![alt text](image-6.png)

Okay, so...hm. I do not find this to be an easy chart to read. What's the story here? It looks like it has increased, but how much? Is there a dip in the middle there? What can we say about the overall trend, other than "it looks like it's going up"? What about monthly seasonality, is that affecting how this looks?

You've probably had to deal with this before. The most common tool for dealing with this in practice is to smooth the time series with a moving average, or something similar. And it's a good solution!

# Smoothing is the most common solution, and it's a pretty good one

The idea of a moving average is simple: average each point with the ones near it in order to make the curve smoother. Sudden spikes will get smoothed out as we combine them with their neighbors. By making the window size 12, we are attempting to remove the 12 month cycle and noise, leaving just the overall long-term trend.

```python
plt.title('Monthly Ridership on the (L) train with smoothing')
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(millions))
plt.plot(l_train_peak_df.num_passengers, marker='o', markersize=12, color='grey', label='Monthly Riders')
plt.plot(l_train_peak_df.num_passengers.ewm(alpha=0.3).mean(), color='black', linestyle='dashed', label='Exponentially Weighted Mean')
plt.plot(l_train_peak_df.num_passengers.rolling(window=12, center=True).mean(), color='black', linestyle='dotted', label='Rolling average')
plt.plot(l_train_peak_df.num_passengers.rolling(win_type='triang', window=12, center=True).mean(), color='black', linestyle='solid', label='Rolling average (Triangular weights)')
plt.legend()
plt.show()
```

![alt text](image-7.png)

That is an improvement, admittedly. It's now much easier to read, for example, that ridership started around the 3 million-ish mark and as of mid-2026 is around the 4 million-ish mark.

This solution hints that there is a smooth trend underlying the data, something like

$$Data_t = Trend_t + Month_t + Noise_t$$

or maybe if you're feeling really fancy, you might write it with Greek letters:

$$\underbrace{y_t}_\textrm{Observed} = \underbrace{\mu_t}_\textrm{Trend component} + \underbrace{\beta_t}_\textrm{Monthly component} + \underbrace{\epsilon_t}_\textrm{Noise component}$$

(For what it's worth, I recommend the Greek letter version. People are always very impressed when data scientists use Greek letters.)

We're removing the monthly cycle and the noise with our moving average, ideally leaving just the trend behind. This is a good trick, but it always feels a little brittle to me. If we had more than one seasonal cycle (for example, daily and monthly cycles), we'd have to do some work to extend our method.

It also raises a natural question - can I see the other parts of the equation, ie the monthly cycle and the noise? They might have information I could use too.

# Get the full picture by decomposing the time series with (M)STL

We can decompose the time series into its pieces using the [MSTL procedure](https://www.statsmodels.org/devel/generated/statsmodels.tsa.seasonal.MSTL.html) as [implemented in statsmodels](https://www.statsmodels.org/devel/generated/statsmodels.tsa.seasonal.MSTL.html).

MSTL stands for "Multiple Seasonal-Trend decomposition using LOESS", which is admittedly a bit of a mouthful. The method decomposes the **observed time series** into three components:

* **Trend:** The smooth long-term trajectory of the time series, with seasonal factors and noise removed.
* **Seasonal:** The cyclic behavior - in our example of month-level data, this is the 12 month cycle. The **M** in in **M**STL comes from the fact that it supports multiple seasonalities (ie, daily _and_ monthly).
* **Residual:** Everything left over. In theory, this is just normally distributed noise. In practice, examining the residuals helps us understand where the model doesn't fit well. 

It's easy to calculate this decomposition in statsmodels, and plot the different components:

```python
res = MSTL(l_train_peak_df['num_passengers'], periods=(12)).fit()

res.plot()
plt.show()
```

![alt text](image-8.png)

All of these add up to the time series we actually observed. They are our estimates of the right hand side of the equation for an additive time series we had before:

$$\underbrace{y_t}_\textrm{Observed} = \underbrace{\mu_t}_\textrm{Trend component} + \underbrace{\beta_t}_\textrm{Monthly component} + \underbrace{\epsilon_t}_\textrm{Noise component}$$

Let's look at each component separately and see what we can learn.

### Trend component

Here's the observed time series, plus the smooth long-term trend component:

```python
# Trend view

plt.title('Monthly Ridership on the (L) train with MSTL trend component')
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(millions))
plt.plot(l_train_peak_df.num_passengers, marker='o', markersize=12, color='grey', label='Monthly Riders')
plt.plot(res.trend, color='black', linestyle='dashed', label='Trend component')
plt.legend()
plt.show()
```

![alt text](image-9.png)

This makes things clearer - it gives us a single smooth line for the trend. We can see a dip in 2024, followed by a recovery over '25 and '26. 

### Seasonal component

```python
# Seasonal view

plt.title('Monthly Ridership on the (L) train - MSTL seasonal component only')
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(millions))
plt.plot(res.seasonal, marker='o', markersize=10)
plt.axhline(0, color='grey', alpha=0.5)
for i in [0, 12, 24, 36]:
    plt.axvline(l_train_peak_df.index[i], linestyle='dashed', alpha=0.5, color='grey')
plt.show()
```

![alt text](image-10.png)

What does this tell us about the seasonal cycles of L train usage?
* October is busiest month, a fact that is well known ([see the NYT for more](https://www.nytimes.com/2013/11/21/nyregion/in-october-a-day-for-the-new-york-city-subways-ridership-record-book.html)) .
* September, on the other hand, is a very low ridership month.
* February tends to be a big dip after January - but it's worth keeping in mind that February has fewer days, which is probably part of the explanation (the bitter cold of the subway in February probably doesn't help).

Comparing the seasonal and trend components using `res.seasonal/res.trend` is interesting too - that will show you how large the "seasonal effect" is compared to the long-term trend.

### Residual component

Lastly, lets look at the residual component. The residuals tell us where a smooth trend + monthly cycle _don't_ account for what's measured. Residuals much larger than the rest are where the model is "surprised" by a sudden change in some month. If the model captures all the relevant structure, then the residuals will look like a normally distributed fuzz around zero.

```python
# Residual view

plt.title('Monthly Ridership on the (L) train - MSTL residual component only')
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(millions))
plt.plot(res.resid, marker='o', markersize=10, linewidth=0)
plt.axhline(0, color='grey', alpha=0.5)
for i in [0, 12, 24, 36]:
    plt.axvline(l_train_peak_df.index[i], linestyle='dashed', color='grey', alpha=0.5)
plt.show()
```

![alt text](image-11.png)

The model seems to fit poorly around the start of 2025; the ridership in January is way higher than the model's estimate. What could have happenened in January 2025 that suddenly caused ridership to jump?

We don't have to look far for an explanation - [on January 5, 2025, New York City implemented its congestion pricing program](https://en.wikipedia.org/wiki/Congestion_pricing_in_New_York_City), a toll on driving in Manhattan south of 61st street. It seems reasonable to conclude that as the L train serves Manhattan and Brooklyn, some drivers chose to substitute a drive into the city with a ride on the L train.

# Which train has increased its ridership the most?

One thing that's nice about this sort of decomposition is that it lets you analyze the trend series, or multiple trend series, to understand what has been happening over the long term. For example, we can ask a question like: **Which subway line has increased monthly ridership the most since 2023?** Using the trend line helps us isolate long-term changes in each series, without worrying that we're accidentally looking at transient effects or the result of seasonal variation.

We can plot the trend-only views of all the lines:

```python
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(millions))
plt.title('Change in Monthly ridership from Jan 2023')

rows = []

for line, color, bullet in [('JZ', 'brown', '🟤'), 
                    ('B', 'orange', '🟠'), ('D', 'orange', '🟠'), ('F', 'orange', '🟠'), ('M', 'orange', '🟠'), 
                    ('L', 'gray', '🔘'), 
                    ('1', 'red', '🔴'), ('2', 'red', '🔴'), ('3', 'red', '🔴'), 
                    ('4', 'green', '🟢'), ('5', 'green', '🟢'), ('6', 'green', '🟢'), ('G', 'lightgreen', '🟢'),
                    ('7', 'purple', '🟣'),
                    ('A', 'blue', '🔵'), ('C', 'blue', '🔵'), ('E', 'blue', '🔵'),
                    ('N', 'olive', '🟡'), ('Q', 'olive', '🟡'), ('R', 'olive', '🟡'), ('W', 'olive', '🟡')]:
    line_peak_df = df[df['line']==line]
    line_peak_df.index = pd.to_datetime(line_peak_df.month)
    
    res = MSTL(line_peak_df['num_passengers'], periods=(12)).fit()
    change_in_ridership = res.trend - res.trend[0]
    plt.plot(change_in_ridership, color=color)
    plt.text(line_peak_df.index[-1], change_in_ridership[-1], line, color=color)
    
    absolute_change = str(round((res.trend.iloc[-1] - res.trend.iloc[0])/1000000, 3)) + 'M'
    percent_change = str(100*round(res.trend.iloc[-1] / res.trend.iloc[0] - 1, 3)) + '%'
    rows.append([line+' '+bullet, absolute_change, percent_change])

print(pd.DataFrame(rows, columns=['Line', 'Absolute Change', 'Percent change']).to_markdown())
```

![alt text](image-12.png)

That's pretty clear - the 6 train is the clear winner! Its trend increased more or less continuously from 2023 to right now in 2026. We could also consider percent growth, as of course subway lines all have different levels of normal traffic.

Okay, that's a little bit busy. Let's look at the summary table for both absolute and percent growth:

|    | Line   | Absolute Change   | Percent change      |
|---:|:-------|:------------------|:--------------------|
|  0 | JZ 🟤  | 0.413M            | 28.6% |
|  1 | B 🟠   | 0.641M            | 23.4% |
|  2 | D 🟠   | 0.351M            | 13.1% |
|  3 | F 🟠   | 1.135M            | 24.3%               |
|  4 | M 🟠   | 0.87M             | 39.9% 🏆 |
|  5 | L 🔘   | 0.908M            | 28.6% |
|  6 | 1 🔴   | 0.817M            | 15.4%               |
|  7 | 2 🔴   | 0.459M            | 12.8%               |
|  8 | 3 🔴   | 0.282M            | 10.4%               |
|  9 | 4 🟢   | 0.944M            | 26.0%               |
| 10 | 5 🟢   | 0.765M            | 25.1%               |
| 11 | 6 🟢   | 1.503M 🏆           | 29.5%               |
| 12 | G 🟢   | 0.277M            | 17.7%               |
| 13 | 7 🟣   | 0.234M            | 4.6%                |
| 14 | A 🔵   | 1.289M            | 31.1%               |
| 15 | C 🔵   | 0.324M            | 16.0%               |
| 16 | E 🔵   | 0.644M            | 15.3% |
| 17 | N 🟡   | 0.798M            | 31.3%               |
| 18 | Q 🟡   | 0.552M            | 16.3%               |
| 19 | R 🟡   | 1.038M            | 25.8%               |
| 20 | W 🟡   | 0.282M            | 18.8%               |

The big winners are the 6 train and the M train! The M is one of my local trains out here on the Brooklyn side, good work lads. The 6 train even got called out in the [Governor's Yearly MTA highlights](https://www.governor.ny.gov/news/governor-hochul-highlights-record-breaking-year-performance-and-ridership-mta-2025) as the train with the most ridership!

# Downsides of MSTL

We've seen lots of ways MSTL is useful in getting a closer look at time series data. What are some of the downsides of this approach?
* The model is flexible enough to add multiple seasonality types on top, but modeling exogenous shocks isn't easy - we can't, for example, easily insert a "congestion pricing" term.
* This method doesn't give us any standard errors around our values. This is unfortunate and a pretty big downside, and isn't trivial to fix because of the autocorrelation of the time series (ie, you can't just do a bootstrap, though maybe you could use a block bootstrap).

MSTL is a great way to understand the different components of the time series, but if these downsides are a problem for your use case, you may want to consider a more powerful modeling approach (like an ARIMA model).