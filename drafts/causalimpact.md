---
layout: post
title: "Causal Inference"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: causalimpact.png
---

# Causal inference is everywhere you look

Despite its lofty and technical name, causal inference seems to come naturally to us. Children are constantly trying things out - with no prompting to do so, you'll find them messing with everything in grabbing range - pets, toys, pots, pans, sounds, etc. In addition to the pleasure of causing chaos, they are often learning along the way, discovering by throwing things about that some objects bounce and other objects shatter. When data scientists engage in similar activity, we usually call it "causal inference", if only for the benefit of our own dignity. Even creatures more intelligent than data scientists (dolphins, cats, etc) are known to engage in similar behavior, learning how things work by trying them out and observing the consequences. This behavior comes so easily to us because understanding cause and effect is core to how we make decisions and navigate the world. 

More sophisticated examples of inferring cause and effect show up in our lives all the time. Americans recently had to evaluate the actions of the late Biden administration, and we spent a lot of time talking about whether the Biden team's decisions produced good effects or bad ones. For example, most Americans were concerned about inflation; had the administration taken the right steps to deal with it? For example, had the very clearly named [Inflation Reduction Act](https://en.wikipedia.org/wiki/Inflation_Reduction_Act) actually done what it claimed to do? Looking at a graph of inflation over time, it seems plausible at first glance:

[Add a plot of inflation over time, with a dotted line for the IRA]

Looking at this chart, we see that between A and B, the inflation rate came down from X to Y. Does that mean that the IRA caused inflation to drop by Y - X points?? If so, that would be pretty remarkable.

The election is over, so it is of course the perfect time to consider this question. Unfortunately, I'm not an economist, so frankly I'm not the ideal person to answer it. But here in the United States we have a grand old tradition of non-experts trying their hand at something new and getting embarrassingly out of their depth, so it is my civic duty to try.

There's another reason I picked this question. In theory, at least, the country in which I reside is a democracy (the precise status of said democracy is currently a subject of lively debate, but we'll save that for another day). 

Googling for about a second indicates that a [handful of economists](https://apnews.com/article/biden-inflation-reduction-climate-anniversary-9950f7e814ac71e89eee3f452ab17f71) asked said that it did not affect inflation, citing some of the factors like the price of energy as the real drivers. 

# How would we know if the Inflation Reduction Act decreased inflation?

"Did the IRA decrease inflation?" is actually a pretty complex question, though it looks simple at first glance. How would we know whether the IRA caused deflation to decrease? One way to try and make sense is to think of it as a counterfactual scenario - what would have happened to inflation if the IRA had **not** been enacted and nothing else changed? If we knew how this counterfactual scenario would have played out, we can compare it to the factual scenario - ie, our universe - and see how big the difference is. 

In causal inference-speak, we call these two scenarios (the inflation rate in each of the two scenarios) the potential outcomes. If we want to think about the IRA's effect on interest rate on day $t$, then

$$\underbrace{\Delta_t}_{Effect} = \underbrace{y_t^1}_{Observed} - \underbrace{y_t^0}_{Not \ Observed}$$

If we had the power to step into an alternate universe we could observe the effect directly

The fundamental problem of causal inference but we don't

DAG 

Ignorability

data 

[Data sheet](https://docs.google.com/spreadsheets/d/1qZFvY9ZGbEC3nX3LvTgdTOtbOdXdrKb8y0_R8Fs-Ufc/edit?usp=sharing)

# Computing a counterfactual with `CausalImpact`

https://github.com/WillianFuks/tfcausalimpact

https://www.youtube.com/watch?v=GTgZfCltMm8

The following code shows no effect; what's missing? Canada's inflation rate, maybe

https://www.bankofcanada.ca/rates/price-indexes/cpi/

https://www.weforum.org/stories/2022/06/inflation-stats-usa-and-world/

Link paper here; you could probably do a plausible version of it on your own with an AR-X model

```python
import pandas as pd

df = pd.read_csv(r'C:\Users\louis\Downloads\CausalImpact Demo - Inflation Reduction Act - Joined dataframe.csv')

import causalimpact

clean_df = df.rename({'US Inflation rate': 'y', 
                      'Gasoline Spot price': 'x1',
                      'Fed Interest rate': 'x2',
                      'GSCPI': 'x3',
                      'Canada inflation rate': 'x4',
                      'UK inflation rate': 'x5'}, axis=1)
clean_df = clean_df[['y', 'x1', 'x2', 'x3', 'x4', 'x5']]

impact = causalimpact.CausalImpact(
    data=clean_df,
    pre_period=[0, 213],
    post_period=[214, 240])

print(impact.inferences['post_preds_means'].dropna())

impact.plot()
```

This tells a very different story from our initial estimate; the effect is not statistically significant, and even the point estimate is ... instead of ... points

# Stress testing our counterfactual model: Placebo trials and assumption robustness

Important assumptions:
* DAG is correct
* More specifically none of the things we control for are downstream of the treatment variable in the DAG
* No confounding, ie the IRA was the only thing that acted on inflation which is clearly incorrect; what else?

Placebo

Sensitivity

# Causal inference is everywhere you look, and now you get to do it too

# Draft ideas

CausalImpact - https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41854.pdf

TFP version in Python - https://github.com/google/tfp-causalimpact/blob/main/docs/quickstart.ipynb

Generate counterfactual via time series modeling, fit on pre-treatment period

y_t = (linear?) local trend + local level from seasonal state space model + f(covar)

Local level is something like AR(1) plus lagged terms for seasonal components

Rubin perspective: We are filling in the other half of the potential outcome equation

Pearl perspective: We are controlling for cyclic effects, the same things that affect parallel time series, and any other factors we include as covariates

# Intro: What and why in causal inference

An example we'll return to again and again: Did the inflation reduction act reduce inflation? (I am not an economist, but please see https://apnews.com/article/biden-inflation-reduction-climate-anniversary-9950f7e814ac71e89eee3f452ab17f71, who cite the price of oil and gas on the world market, interest rate, and supply chain challenges as causes)

Potential outcomes

DAG

# Problem setup

Break down the model - functional form, dag

# Using CausalImpact to estimate the effect

# Doing it right: Placebo, Sensitivity

# Outro

Some last thoughts

Are your counterfactual states well defined? Is it really clear whether "more jobs" is an unalloyed good?
