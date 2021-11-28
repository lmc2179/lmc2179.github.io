---
layout: post
title: "How active are my most active users? My least active ones? Quantiles and their confidence intervals in Python"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: bad_day.png
---

*Analysts spend a lot of time thinking about the mean, which is by far the most common descriptive statistic we extract. But looking at the average observation leaves out a lot of information about the shape of the distribution . Comparing the shapes of two distributions . Example*

# Example intro

Number of newsletter opens - 

Possible distributions with the same mean (poisson, skellam, nbinom, uniform, geom(1./50)

Similar examples: Anomaly detection (if the process is serially uncorrelated)

[Zombo.com newZletter](https://www.zombo.com/join1.htm)

# Quantile idea and point estimate

Histogram

Pandas describe

IQR, Box and whisker

# Quantile inference uncertainty

MJ standard errors

CDF/Quantile curve

# Two-sample quantile comparison

Two sample histograms

Quantile curve with uncertainty bands, significant changes marked\

# Outro: Quantile regression

Mean is to linear regression as quantiles are to quantile regression

# Appendix: Simulated coverage of the quantile's SEs

The method works in a simulation similar to one that might have generated our data; note where coverage is good and where it is not

# Appendix: Alternative ways of getting the standard errors

Sampling distribution with known pdf, Exact method, bootstrap
