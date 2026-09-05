
https://data.ny.gov/Transportation/MTA-Subway-Customer-Journey-Focused-Metrics-Beginn/r7qk-6tcy/about_data 

Title: Finding New York's Hottest Train with time series decomposition

# Lots of very important decisions are made by looking at time series data

A shocking number of real world decisions are made by people on a zoom call squinting at a time series chart of a metric and saying "okay, I think I know what's going on here". But do they? In their defense, a time series can be hard to read. For example, the other day as I was sitting on the subway to go to my office (where I would sit on zoom calls squinting at time series plots), I found myself wondering whether there are more people on the subway than there had been a few years ago. The subway certainly _felt_ more crowded, but maybe I'm just looking through rose tinted glasses at the New York of yesteryear (I wouldn't be the first). Had the subway ridership actually increased? Well, that's easy, we can grab a data set from our good friends at the MTA, and look at train ridership over time. For example, here's the L train, which I was commuting on:

{code: import}

{plot of the L train ridership over time}

Okay, so...hm. I do not find this to be an easy chart to read. What's the story here? It looks like it has increased, but how much? Is there a dip in the middle there? What is the overall trend?

You've probably had to deal with this before. The most common tool for dealing with this in practice is to smooth the time series with a moving average, or something similar. And it's a good solution!

# Smoothing is the most common solution, and it's a pretty good one

The idea of a moving average is simple: average each point with the ones near it in order to make the curve smoother. 

{Picture of moving average results}

That is an improvement, admittedly.

This solution hints that there is a trend underlying the data, something like

$$Data_t = Trend_t + Month_t + Noise_t$$

or maybe if you're feeling really fancy, you might write it with Greek letters:

$$y_t = \mu_t + \beta_t + \epsilon_t$$

$\underbrace{r_t^{UK} - r_{t-1}^{UK}}_\textrm{UK revenue growth}$

(For what it's worth, I recommend the Greek letter version. People are always very impressed when data scientists use Greek letters.)

We're removing the monthly cycle and the noise with our moving average, ideally leaving just the trend behind. This is a good trick, but it always feels a little brittle to me. If we had more than one seasonal cycle (for example, daily and monthly cycles), we'd have to do some work to extend our method.

It also raises a natural question - can I see the other parts of the equation, ie the monthly cycle and the noise? They might have information I could use too.

# Get the full picture by decomposing the time series with (M)STL

We can decompose the time series using MSTL

{L train MSTL results}

All of these add up to the time series we actually observed

Mention the equation again

{L train plus trend}

How does the trend look?

{Seasonal}

Does it tell us anything about the seasonal cycle?

October is busiest, a well documented fact https://www.nytimes.com/2013/11/21/nyregion/in-october-a-day-for-the-new-york-city-subways-ridership-record-book.html 

September is low

February is always a big dip after jan; though it is the shortest

res.seasonal/res.trend is neat too - seasonal effects cause the trend to move around -15-+10 % depending

{Residuals}

What do the residuals look like?

The biggest "surprise" is a much larger ridership in Jan 2025

oh wow Congestion pricing

# Which train has increased its ridership the most?

Show all of the lines

Show them as % growth

Show the top 5

Show the bottom 5

|    | line   |     |
|---:|:-------|:----|
|  0 | JZ 🟤  |     |
|  1 | B 🟠   |     |
|  2 | D 🟠   |     |
|  3 | F 🟠   |     |
|  4 | M 🟠   |    |
|  5 | L 🔘   |     |
|  6 | 1 🔴   |     |
|  7 | 2 🔴   |     |
|  8 | 3 🔴   |     |
|  9 | 4 🟢   |     |
| 10 | 5 🟢   |     |
| 11 | 6 🟢   |     |
| 12 | G 🟢   |     |
| 13 | 7 🟣   |     |
| 14 | A 🔵   |     |
| 15 | C 🔵   |     |
| 16 | E 🔵   |     |
| 17 | N 🟡   |     |
| 18 | Q 🟡   |     |
| 19 | R 🟡   |     |
| 20 | W 🟡   |     |

# Downsides of MSTL

No standard errors - this is a big one . maybe block bootstrap could fix this

Consider building a more complex regression model, esp an ARMA model

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