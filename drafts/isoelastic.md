---
layout: post
title: "Elasticity and log-log models for practicing data scientists"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: important_relationship.png
---

https://en.wikipedia.org/wiki/Isoelastic_function

# it's everywhere

I have spent a shocking percentage of my career drawing some version of this diagram on a whiteboard:

-Important relationship plot-

This relationship has a few key aspects that I notice over and over again:
* The output increases when more input is added; the line slopes up.
* Each input added is less efficient than the last; the slope is decreasing.
* Inputs and outputs are both positive

If you're an economist, or even if you just took econ 101, you likely recognize . Economists seem to be familiar with a lot of the useful properties of this kind of relationship 

Nonetheless, I am not an economist (though I've had the pleasure of working with plenty of brilliant people with economics training). If you're like me, then you might not have these details close to hand. This post is meant to be a small primer for anyone who needs to build models with these kinds of functions

We usually want to know this relationship so we can answer some practical questions such as:
* How much input will we need to add in order to reach our desired level of output?
* If we have some free capital, material, or time to spend, what will we get for it? Should we use it here or somewhere else?
* When will it become inefficient to add more input, ie when will the value of the marginal input be less than the marginal output?

Lots and lots and _lots_ of measured variables seem to have this relationship. The notion of elasticity is ubiquitous in economics, which seems to be the first usage of this kind of model that I can find. However, in my own career I've seen this shape of input-output relationship show up over and over:
* Marketing spend and impressions
* Number of users who see something vs the number who engage with it
* Number of samples vs model quality
* Time spent on a project and quality of result

It also resembles cobb-douglas

# Some useful facts about 

## Algebra

### Functional form

### Percent changes and interpretations of the coefficient

### Inverting the function

## Log-log regression

# code

get data

log log regression

notes about percent changes in a table

# does this model really describe reality? some causal considerations and notes on modeling

The above analysis is, when you get down to it, mostly algebra. It's useful algebra to be sure, but it is really just repeated manipulation of the functional form $\alpha x ^ \beta$. 

# outro

Some other options, esp. w/ zeros:

IHS
https://marcfbellemare.com/wordpress/wp-content/uploads/2019/02/BellemareWichmanIHSFebruary2019.pdf
https://worthwhile.typepad.com/worthwhile_canadian_initi/2011/07/a-rant-on-inverse-hyperbolic-sine-transformations.html

Maybe sqrt

