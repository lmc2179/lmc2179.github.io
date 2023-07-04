---
layout: post
title: "Practical poisson"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: MulberryPhiladelphia.jpg
---

# Count data is everywhere ("Commonality of counting")

We are _constantly_ counting stuff, and collecting data about counts of things that are happening. Some examples that come to mind easily for practicing data analysts:

* Customers showing up on a website
* Leads landing in a sales system
* Transactions being generated on ecommerce website
* Raw material showing up to be refined or used in production
* Count of errors

We care about these counts - are they going up? Are they going down? Did we get more when we tried some new method? Is there a change from last time to this time? Usually, they form the first step in a funnel or production process, and our final output is partly a function of how many inputs we get.

More practically speaking, there are a bunch of handy statistical tools that we can use when we're counting things. Those tools make it easy to calculate standard errors and CIs when we only know the count of something. Those measures of uncertainty will help us separate signal from noise, and make better decisions as we think about those counting processes.

Let me provide a more colorful example (a purple one, it will turn out). So, I recently found out that my backyard has a _very_ productive berry tree. I know sadly little about berries (I thought they only grew on bushes!?), but after some Googling I discovered that my new neighbor was a mulberry tree. Mulberries, it turns out, are edible and quite tasty! Since I had to clean them up anyway, I collected them and started turning them into jam and wine (jam is a success so far; wine is a work in progress). 

Let's ask a practical question now. It rained very hard today yesterday but not today - is there a statistically significant difference in the arrival count between the two days?

This case study is tiny data; literally two numbers. However, if we're willing to make certain assumptions, it will turn out we can answer this question.

# What are the key features of these count data questions

a set period with exponential arrivals - this is empirically measurable

discrete counts of items

# how do we make a formal model of these problems

poisson distribution. horse kick

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

posterior predictive: example, planning for how many berries I might get over the next week

# Also, a normal estimate 

compare also: Normal-based estimate

# going from here

what if it's not poisson? negative binomial, https://en.wikipedia.org/wiki/Overdispersion

afterwards: Rates based on binomial, or perhaps a more interesting function. convert raw product into other discrete products (Binomial) or yield base on average per unit.

conditional means: poisson regression


[Image from wikipedia.](https://commons.wikimedia.org/wiki/File:MulberryPhiladelphia.jpg)