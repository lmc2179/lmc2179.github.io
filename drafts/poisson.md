---
layout: post
title: "Practical poisson"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: MulberryPhiladelphia.jpg
---

# Count data is everywhere ("Commonality of counting")

Idea: event counting is everywere. most funnels or production processes start with the arrival of raw material, then subsequent transformation into other stuff

example: mulberries landing on my deck

# What are the key features of these count data questions

a set period with exponential arrivals

discrete counts of items

# how do we make a formal model of these problems

poisson and exponential relationship proof (there's one on stackoverflow). note about what is a poisson process vs a poisson distribution

important feature of the poisson distribution: mean = variance

model of the treatment effect

# CIs and SEs of the rate

poisson-gamma

Demo of poisson-gamma distribution with "flat" prior. Note coverage is good with a single sample unless rate is near one

```
from scipy.stats import gamma, poisson
import numpy as np


n_sim = 1000

n_sample_size = 1

test_rates = [0.5, 1, 2, 3, 4, 5, 10]

coverage_rates = []

for true_rate in test_rates:
    correct_coverage = 0
    a,b = .00001, .00001
    
    for _ in range(n_sim):
        data = poisson(true_rate).rvs(n_sample_size)
        low, high = gamma(a + np.sum(data), scale = 1./(b + len(data))).interval(.95)
        if low < true_rate < high:
            correct_coverage += 1
    
    coverage_rates.append(correct_coverage / n_sim)
    
plt.plot(test_rates, coverage_rates, marker='o')`
```

# Also, a normal estimate 

compare also: Normal-based estimate

# going from here



afterwards: Rates based on binomial, or perhaps a more interesting function. convert raw product into other discrete products (Binomial) or yield base on average per unit.

conditional means: poisson regression


[Image from wikipedia.](https://commons.wikimedia.org/wiki/File:MulberryPhiladelphia.jpg)
