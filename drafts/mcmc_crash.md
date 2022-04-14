---
layout: post
title: "PyMC3 makes Bayesian Inference in Python Easy"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: test.png
---

# Everything you need to run a Bayesian analysis

I've met lots of folks who are interested in Bayesian methods but don't know where to start; this is for them

For stats people but not Bayesians

Do not focus too much on frequentist/Bayesian diffs

Create a clear Bayesian Model building process; offload technical details to technical references

Why?
* Note access to other methods that it allows, like hierarchical
* Cite pragmatic statistics
* Cite a strong defense of the Bayesian perspective

# What does a Bayesian Analysis look like?

The absolute shortest, quickest, Hemingwayest description of Bayesian Analysis that I can think of is this:

> We have a question about some parameter, $\Theta$. We have some prior beliefs about $\Theta$, which is represented by the probability distribution $\mathbb{P}(\Theta)$. We collect some data, $X$, which we believe will tell us something additional about $\Theta$. We update our belief based on the data, using Bayes Rule to obtain $\mathbb{P}(\Theta \mid X)$, which is called the posterior distribution. Armed with $\mathbb{P}(\Theta \mid X)$, we answer our question about $\Theta$ to the best of our knowledge.

This short synopsis of the Bayesian update process gives us a playbook for doing Bayesian statistics:

1. ?
2. ?

# ?

Examples: Usual normal, reduced normal

# Detour: What happened when we call the sample function

intuitive MCMC

# How do we know it worked

Sampling statistics for diagnosing issues

Model checks

Posterior Predictive

LOOCV?

# Good books on Bayesian Inference

Gelman
Kruschke

# Other cools Bayes-related libraries

emcee
bayes boot
