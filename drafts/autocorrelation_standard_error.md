---
layout: post
title: "Autocorrelation and standard error"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: bad_day.png
---

Little sapling data scientists, coming up in the Garden of Analysis, learn early on that their best friend is the **sample mean**:

$$\hat{\mu} = \frac{1}{n} \sum_{i=1}^{i=n} y_i$$

They also learn that statements about the sample mean are most useful if you calculate the **standard error of the sample mean**

$$\hat{SE}(\hat{\mu})= \frac{\sigma}{\sqrt{n}}$$

This calculation is at the heart of many practical applications, especially, clinical trials and A/B tests. It's a powerful tool, letting practitioners make inferences about populations in a very wide variety of situations.

Central Limit Theorem

Demo sim shows that it works, of course it does

What if we add mild autocorrelation, like 0.1. This is realistic; what about time series metrics, etc. You definitely have seen this

Oh no our coverage rate suddenly doesn't look the way we want

The issue here is that it's not IID. Each data point contains "less information" than the IID case, since it's correlated with the last one

Two options:
* Add lags to the model for as many periods as there is autocorrelation. This makes the \epsilon_t IID again. Con: Requires you to have a pretty good idea of how much autocorrelation there is
* Robust standard errors. Cons: I think they're wider?