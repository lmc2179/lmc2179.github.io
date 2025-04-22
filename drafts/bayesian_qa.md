---
layout: post
title: "Do better QA with just a little bit of Bayesian statistics: Beta-Binomial analysis in Python"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: bayes_neon.jpg
---

# Do better QA with just a little bit of Bayesian statistics

Whatever product you're making - software, drugs, lightbulbs, AI chat experiences, croissants, pulse oximeters, you name it - you eventually will need to figure out whether your product is good enough. This takes many forms:
* Software
* Lightbulbs
* AI Chat
* ?

This process of measuring the good-enough-ness of the product is commonly called Quality Assurance (QA) or perhaps Quality Control (QC). But quality is a famously tricky thing to understand. <sup>[1](#foot1)</sup>

General methods for QA of a product:
* Direct inspection - the most common, and the one we talk about here. Have a person or automated system look at it, and evaluate whether it is good enough
* Study the effects of the products - A/B tests, clinical trials, measure whether it does what you want
* Simulated environment - Cross-validation, physical device testing, unit testing

Direct inspection is usually the first thing you try, especially if you're not sure what you're looking for yet. The problem is that this this is usually expensive, and so we usually can't inspect every outgoing item; we can only inspect a sample. We need to use the sample to decide whether this batch is okay to ship or not; we will base this decision on whether the failure rate for the whole batch is greater than the highest acceptable failure rate

Okay so you've got the basics down: you know your unit, you have your evaluation measure (binary or real), you have a batch, you have a target failure rate

At this point, you've decided on:
* What the unit of evaluation is (one lightbulb, one chat session, etc)
* How you will measure whether it is meets your standards (manual inspection, automated measurement, etc)
* The size of the sample you're going to collect, $n$
* The highest acceptable failure rate, $\mu^{*}$

You collect $y = 40$ out of $n = 1000?$ failures. is that okay? It seems pretty close, how sure are we that the rest of the batch is safe?

# Bayesics: Beta Binomial analysis of the failure rate

Bayesian analysis is a little three-step dance that goes like this:
* Pick a prior
* Observe the data
* Update the prior to get the posterior. Use the posterior to calculate the probability  $\mu < 5\%$.

The beta distribution is a relatively flexible distribution over the interval $[0, 1]$. 

Conjugate prior:

$\underbrace{\mu}_{\text{Our prior on the rate}} \sim \underbrace{Beta(\alpha_0, \beta_0)}_{\text{is a Beta distribution with params } \alpha_0, \beta_0}$

You can interpret the prior parameters as "hypothetical data" that summarizes your beliefs about the rate. 

Posterior:

$\underbrace{\mu \mid y, n}_{\text{The posterior of the rate given the data}} \sim \underbrace{Beta(\alpha_0 + y, \beta_0 + N - y)}_{\text{is given by this beta distribution}}$

That was kind of a lot, so here's a handy little table, which each step of the process

## Doing the analysis for our sample

https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html

```python
from scipy.stats import beta

y = 40
n = 1000

a_0 = 1./3
b_0 = 1./3

posterior = beta(a_0 + y, b_0 + n - y)

n_simulations = 100000
posterior_samples = posterior.rvs(n_simulations)

print('Monte carlo estimate of P(Rate) < 5%: ', sum(posterior_samples < .05) / len(posterior_samples))

print('CDF(5%) = ', posterior.cdf(.05))
````

## Digression: Picking a prior

### "Flat" or "uninformative" priors

_If you're just getting started, the recommended prior of 1/3, 1/3 is probably good enough. But using statistics responsibly does mean thinking through all the details of your method, so pinky promise me you'll come back and read this sometime, okay?_

A commonly used prior is ... . If you think more than one prior, applies, the safest thing is to try all of them and see how much they change your result.

|Values of $\alpha, \beta$ | Fancy name | Notes |
|-|-|-|
|$(0, 0)$|Haldane|Improper prior|
|$(1, 1)$|Uniform|Assigns equal value to all possibilities|
|$(\frac{1}{2}, \frac{1}{2})$|Jeffreys|Jeffrey's rule reference prior|
|$(\frac{1}{3}, \frac{1}{3})$|Neutral|Ensures median|

Kerman paragraph 

See [Kerman 2011](https://projecteuclid.org/journals/electronic-journal-of-statistics/volume-5/issue-none/Neutral-noninformative-and-informative-conjugate-beta-and-gamma-prior-distributions/10.1214/11-EJS648.full).

### Informative priors based on expert knowledge

Certain posteriors are just not possible, and we should represent this fact in our prior

# Extrapolating to the whole batch: Predictive simulation with the Beta-Binomial model

We often want to know how many defective are in the whole batch for planning (maybe we make a warantee for customers, or have support centers which need to be staffed, or expect other adverse effects for each one shipped)

https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.betabinom.html

# What's the smallest sample I could collect?

Sample size -> Precision is the real thing. Simulation?

# Where to learn more

Gelman

## Endnotes

<a name="foot1">1</a>: _Zen and the Art of Motorcycle Maintenance_ explores this in a way which I found to be amusing, if not exactly sound philosophical material.

-----------------------------------
-------------------------
------------------------------
------------------------------------
--------------------------------

# Draft

# Quantity ☯ Quality

if set our sights on quality alone we become perfectionists; if we aim for pure quantity we will produce a large volume of low-value output

It's often easy to measure quantity so metrics usuall do; it's usually more expensive to measure quality

but we need to

Often, we can only measure the quality of a subset of outputs due to the cost, so we need to do some statistics

luckily its easy

Definitions: What is a unit of output, what is a "good enough quality" unit

# A touch of statistics makes QA much easier

QA questions related to estimating the number of defective units pop up all the time, even outside the manufacturing context

QA is expensive, so we should be careful about how much of it we do

examples:
* factory
* data quality checks in a data warehouse
* checking the quality of your ml model vs human judgement

# basics: normal interval

sometimes assume the worst case of p=0.5

# Estimating the proportion of defects in a finite population: The finite population correction

Finite population correction

```python
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import sem

x = np.array([1]*3000 + [0]*3000)

n_population = len(x)

n_sample = 1000

n_sim = 10000

fpc = np.sqrt((n_population - n_sample) / (n_population - 1))

samples = [np.random.choice(x, size=n_sample, replace=False) for _ in range(n_sim)]
sampled_means = [np.mean(s) for s in samples]
sampled_sems = [sem(s) for s in samples]
sampled_corrected_sems = [sem(s) * fpc for s in samples]

actual_sem = np.std(sampled_means)

sns.distplot(sampled_sems)
sns.distplot(sampled_corrected_sems)
plt.axvline(actual_sem)
plt.show()

sns.distplot(sampled_sems - actual_sem)
sns.distplot(sampled_corrected_sems - actual_sem)
plt.axvline(np.mean(sampled_corrected_sems - actual_sem))
plt.axvline(np.mean(sampled_sems - actual_sem))
plt.show()

```

Plot (n / N) vs standard error inflation (Inverse of the correction factor?)
Swap X with a different distribution
Is this exact for the binomial distribution!? Note that the correct SE is correct event for tiny sample sizes

Bayesian view: Predictive posterior for the unobserved values for binomial (exact I think), else bootstrap unknown values
(Draw a beta, then draw a binomial - so beta binomial)

# Prediction the number defective units based on the sample: the Beta-Binomial model

```python
from scipy.stats import betabinom, beta, binomial

betabinom(a, b, n).rvs(k)

binom(n, beta(a, b).rvs(k)).rvs() # Same deal

```

# A shortcut when defects are rare events: The rule of three

compare with beta interval using jeffrey's prior

