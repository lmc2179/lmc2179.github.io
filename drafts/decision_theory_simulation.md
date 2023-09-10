---
layout: post
title: "How not to run out of gin: Robust decision-making under uncertainty using simulations"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: gin_recipe.png
---

The reason we look at data is so we can make better decisions. However, even after consulting the data, there is still usually some uncertainty what will happen. We want to make choices that will be effect in not just the average situation, but whatever range of outcomes are plausible. In this post, we'll cover some basic Decision Theory and use it to show how simulations can help us plan for situations when uncertainty is present.

_**Note from the Editors**: The Editorial team here at `Casual Inference: Data Analysis and Other Apocrypha™` remind you to be responsible when drinking and calculating expected values. Alcohol may impair your ability to operate a Monte Carlo Simulation or pronounce the word "heteroskedasticity". The Editorial Board does not endorse any of the views, recipes, or probability calculations present in this article, which are solely those of the author._

# Start by describing the situation

Most of the time when we make a decision, the outcome of our decision isn't guaranteed. Life is complicated, and our knowledge of what will happen is imperfect. I deal with a prototypical example of this every time I step out to catch the train to work - should I bring an umbrella with me? It's annoying to carry one if it isn't necessary, but it's much more annoying to get caught on the rain as I scramble off the J train to try and make my first meeting of the morning (I really need to start leaving earlier). What should I do? An exhaustive analysis proves frustrating: if I bring the umbrella, I _might_ not need it , but if I don't bring it I _might_ also get caught in the rain. So are both choices bad ones?

It seems unlikely that both choices are _equally_ bad. After all, on most days I can usually make it past the umbrella rack without having a nervous breakdown, so there must be some more information for us to use. There are two important things to note about this choice:
* **The decision relies on certain unknown facts, ie whether it will be raining when I get off the train. But we usually have estimated probabilities about those facts obtained from data and expertise**. We usually have an estimate of the probability of rain - either a numerical one from the Weather app on my phone, or a rougher one obtained by squinting at the clouds and making a guess.
* The worst cases are very different: Needing to carry an uncesessary umbrella is an annoyance, but getting caught in a downpour can make the rest of your day much more difficult. **The possible consequences of my decision vary in how bad (or good) they are**.

Leonard J. Savage's _The Foundations of Statistics_

I'm going to replace the unwieldy "state of the world" with "scenario" in this writeup, just since I've found that easier to get across to people. Plus, then you can tell your MBA-wielding VP that you did a [scenario analysis](https://en.wikipedia.org/wiki/Scenario_planning), which I think you'll agree sounds very business-y.

So we can break our decision into pieces
* Scenario: weather when you get off the train - rain or no rain
* Action: go without umbrella, go with umbrella. made without necessarily knowing all the information.
* Consequence: got wet or didn't

Consequences = Scenario-Action matrix

Find action which maximizes expected value (some alternatives are possible, discussed later)

show a little mathematical formulation of this

# A real life analysis: How much raw material do we need to satisfy demand?

Most problems are more complex than this, and it is often easier to simulate than to solve. really then this is a monte carlo method

some realistic aspects: estimatimg inputs required for demand; lots of uncertainty; complicated process and multi-dimensional;

defining the SAC

running the simulations

picking one that doesn't fail

estimating the level of waste for our chosen decision

# Where do the scenario parameters come from?

elicit from expertise

predictive models

previous experiment results

bayesian analysis

check them against historical data and how well your assumptions matched with reality
