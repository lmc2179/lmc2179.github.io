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

Let's try and organize the situation so we can analyze the outcomes of the choices we have. We're going to use the framework laid out in Leonard J. Savage's _The Foundations of Statistics_. In this framework, we model the decision as:
* The **Actions** which we can take. In this case, take an umbrella or don't.
* The **States of the World, or Scenarios** which might end up being true. In this case, whether it is raining or not when I get off the train.
* The **Consequences** of our action given the scenario. The results of the action we chose, plus the scenario together - getting rained on, having to carry an umbrella, etc.

I'm going to replace the unwieldy "state of the world" with "scenario" in this writeup, just since I've found that easier to get across to people. Plus, then you can tell your MBA-wielding VP that you did a [scenario analysis](https://en.wikipedia.org/wiki/Scenario_planning), which I think you'll agree sounds very business-y.

It's often handy to write out the "consequences" part of the model as a scenario $times$ action matrix:

|     | **Take umbrella** | **Don't take** |
| -------- | ------- | ------- |
| **No Rain**  | Mild inconvenience  🫤  | Status Quo |
| **Rain**     | Status Quo    | Caught in downpour 😢 |



Eagle-eyed readers will recognize this as a looking a lot like the confusion matrix, which we've [applied to optimal decision-making before](https://lmc2179.github.io/posts/decisions.html). In that case, we realized that if we know the probability distribution over the rows of the matrix, we can compute the expected value of each action. If we were to assign each of the consequences a score to create a score matrix, then we could just calculate

$$\mathbb{E}[Score | Action] = \sum_i Score[Scenario \ i, Action] \times \mathbb{P}(Scenario \ i)$$

And pick the action with the larger expected value. This is already pretty useful! However, sometimes we want more visibility into what kind of outcome we get. Perhaps the outcome space is difficult to summarize with a single number, for example.

# A real life analysis: How much raw material do we need to satisfy demand?

Most problems are more complex than this, and it is often easier to simulate than to solve. really then this is a monte carlo method

some realistic aspects: estimatimg inputs required for demand; lots of uncertainty; complicated process and multi-dimensional;

defining the SAC

running the simulations

picking one that doesn't fail

estimating the level of waste for our chosen decision

# A practical checklist for building decision simulations

Where do the scenario parameters come from?
* elicit from expertise
* build a predictive model of the distributions/conditional distributions
* previous experiment results
* bayesian analysis
check them against historical data and how well your assumptions matched with reality
