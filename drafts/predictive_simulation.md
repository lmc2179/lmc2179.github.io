---
layout: post
title: "Predictive Simulation of Time Series with ARIMA in Python"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: ???.jpg
---

# We build forecast models so we can make plans and decisions about the future

>**Caesar:** Who is it in the press that calls on me?
I hear a tongue, shriller than all the music,
Cry 'Caesar!' Speak; Caesar is turn'd to hear.
>
>**Soothsayer:** Beware the ides of March.

~_Julius Caesar_ receives a times series forecast from his data scientist in Act I, Scene 2

Every person and organization needs to have a plan, but planning is hard because we're not sure what will happen. How much demand will there be for our product this quarter? How many people will come to our coffee shop? If we could answer these questions, we could figure out how many people to hire, how many products we'll need to make, how much coffee we should buy, etc.

Of course, people still find ways to muddle through. Generally, we make think about the likely scenarios, and plan for those. If you possess some data about the past, you can use it to make a model, and the model can tell you what scenarios you should plan for.

If you've ever lived in our near a hurricane zone (or for that matter, seen a hurricane on the news), then you've probably seen a [spaghetti plot](https://en.wikipedia.org/wiki/Spaghetti_plot). It shows all the paths that the hurricane might take, based on meterological analysis:

![Hurricane dorian spaghetti plot](image-2.png)    
_Spaghetti plot of Hurricane Dorian. Each strand is a path that a simulated hurricane took in one of many runs of the simulation._

This is a tool for planning about an uncertain future. If a hurricane is coming to your town, you had best be ready. Organizations like FEMA need to figure out how to deploy personnel, supplies, potential evacuation orders, and other high-stakes decisions. Residents of those places need to understand how likely it is that the hurricane will appear near them. 

The spaghetti plot is a very rich diagram! First, it shows us what sort of overall path the hurricane is likely to take. Secondly, it lets us look up a specific spot on the map, and see how likely it is that a hurricane will pass through, by looking at how densely packed the paths are.

There are plenty of decisions in my life (professional and personal) where a spaghetti plot would have come in handy. 

```python
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv')

y = df.Passengers
n_obs = len(y)

train_cutoff = 96
validate_cutoff = 120

from matplotlib import pyplot as plt

plt.plot(y)
plt.axvline(train_cutoff, color='orange', linestyle='dashed')
plt.axvline(validate_cutoff, color='green', linestyle='dashed')
plt.legend()
plt.title('Airline passengers by month')
plt.ylabel('Total passengers')
plt.xlabel('Month')
plt.show()

```

What happens next? If we know, we could plan: Supplies, personnel, cost, rev, etc

Predicting where a time series will go next can be done with a model like the AR model we discussed last time. that model gave us E[y_t] and error bands, which can be used

We'd ideally like to see a sample of all the future paths the time series could take - _predictive simulation_, which shows us the possible futures we might need to plan for. when we do this for hurricanes we call them spaghetti plots (wiki page)

Big picture, what we'll do is:

1. Collect time series data
2. Fit a time series model, and demonstrate goodness of fit
3. Simulate future paths to see what might happen.

We'll use statsmodels SARIMAX (link to it) and just the ARIMA part

# What is all this alphabet soup? Breaking down the ARIMA and SARIMAX models

Today, we're going to use an ARIMA model, so we'll start with that. An ARIMA model is the sum of the two ARMA components

Autoregerssige (AR) component: y_t is a function of previous observations. link post

The AR model is

MA - $y_t$ is a function of previous innovations (?) Link to AR/MA comparison link

The MA model is

Sometimes, it makes  more sense to predict the difference from one y to the next in which case

I - Predict $\Delta y_t$ instead; differencing

See https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average#Examples for some intuition on how this relates to other stuff like random walks

That means that an ARIMA model has 3 free parameters

The same class also supports SARIMAX, which has the ARIMA components plus some more of them.

S - Cyclic version of ARIMA

SARIMAX = S + ARIMA + X

X - Not used, link to post and note relevant kwarg

# Model selection and cross-validation

We want to select a model and demonstrate prediction quality

Three-way CV split: (1) Train (2) Param. select (3) Sim quality

(1) Code for training

(2) Random search - Bergstrom

```python
# Random search for hyperparameters
n_hyperparams_to_test = 300
max_hyperparam_val = 12

results = []
for i in tqdm(range(n_hyperparams_to_test)):
    params = np.random.randint(0, max_hyperparam_val+1, size=3)
    p, d, q = params
    model = sm.tsa.statespace.SARIMAX(y[:train_cutoff], order=(p, d, q), trend='ct')
    try:
        model_fit = model.fit(disp=False)
    except:
        error = np.inf
    else:
        y_test_predicted = model_fit.forecast(validate_cutoff - train_cutoff)
        y_test_actual = y[train_cutoff:validate_cutoff]
        error = np.mean((y_test_actual-y_test_predicted)**2)
    results.append([error, params])
    
cv_df = pd.DataFrame(results, columns=['MSE', 'Parameters']).sort_values('MSE')
best_params = cv_df['Parameters'].iloc[0]
```

The best params are

Okay, lets show the train and test fit

```python
# Check: Plot best model against train and test region
p, d, q = best_params
model = sm.tsa.statespace.SARIMAX(y[:train_cutoff], order=(p, d, q), trend='ct')
model_fit = model.fit(disp=False)

plt.plot(y)
plt.plot(model_fit.predict(end=validate_cutoff))
plt.axvline(train_cutoff, color='orange', linestyle='dashed')
plt.axvline(validate_cutoff, color='green', linestyle='dashed')
plt.legend()
plt.title('Airline passengers by month')
plt.ylabel('Total passengers')
plt.xlabel('Month')
plt.show()
```

(3) Demonstration of sim quality, ie coverage of real path of validation set

```python
# Fit a SARIMAX model as an AR(2) model (order=(2, 0, 0))
model = sm.tsa.statespace.SARIMAX(y, order=(10, 1, 5))
model_fit = model.fit(disp=False)
print(model_fit.summary())

# Set simulation parameters
n_forecast = 50
n_simulations = 100

# Container for simulated paths
simulations = np.empty((n_simulations, n_forecast))

# Simulate multiple paths. We use a loop to generate each simulation.
for i in range(n_simulations):
    simulations[i, :] = model_fit.simulate(nsimulations=n_forecast, anchor='end')

# Plot observed data
plt.figure(figsize=(12, 6))
plt.plot(np.arange(n_obs), y, label='Observed Data', color='black')

# Plot each simulated path
for i in range(n_simulations):
    plt.plot(np.arange(n_obs, n_obs+n_forecast), simulations[i, :],
              label=f'Simulation {i+1}', alpha=0.1, color='blue')

plt.xlabel('Time')
plt.ylabel('Value')
plt.title('SARIMAX Model: Observed Data and Multiple Simulated Future Paths')
#plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()
```

Code - Path coverage

Pct coverage? Narrow enough to be useful?

# Training the final model and generating predicted paths

Okay, so we've selected the model (., ., .). Let's train it on the whole data set:

Code - Fitting the combined model

Code/Summary - Interpreting the combined model



Now we can make predictions. Generating simulated paths for future flight counts

Code - Path samples, 99% quantiles

What do we find out? Upper and lower limits

When can we interpret this causally? When we have closed all the backdoors

# Monte Carlo analysis of the simulated paths

For example: Revenue assuming x% market share