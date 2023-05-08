---
layout: post
title: "How to stop an experiment early with alpha spending"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: alpha_spending.png
---

Let me tell you a story - perhaps a familiar one.

> **Product Manager**: Hey `$data_analyst`, I looked at your dashboard! We only kicked off `$AB_test_name` a few days ago, the results look amazing! It looks like the result is already statistically significant, even though we were going to run it for another week.
>
>**Data Analyst**: Absolutely, they're very promising!
>
>**Product Manager**: Well, that settles it, we can turn off the test, it looks like a winner.
>
>**Data Analyst**: Woah, hold on now - we can't do that! 
>
>**Product Manager**: But...why not? Your own dashboard says it's statistically significant! Isn't that what it's for?
>
>**Data Analyst**: Yes, but we said we would collect two weeks of data when we designed the experiment, and the analysis is only valid if we do that. 
>
>I have to respect the arcane mystic powers of ✨`S T A T I S T I C S`✨!!! 

_**Has this ever happened to you?**_

This is a frustrating conversation for all involved. The PM is trying to play by the rules by looking at the significance of the test

# The issue with stopping early

# Some quick fixes

bonferroni

linear spending

# Solving the problem with the OBF Alpha spending function

https://eclass.uoa.gr/modules/document/file.php/MATH301/PracticalSession3/LanDeMets.pdf

```
def obf_alpha(t_prop): return 2 - 2*(norm.cdf(1.96/np.sqrt(t_prop)))
```

# A note on coverage rates
