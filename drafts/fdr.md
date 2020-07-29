---
layout: post
title: "An alternative to the FWER for multiple comparisons: The False-Discovery Rate"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: jellybeans.png
---

*We've [previously explored](https://lmc2179.github.io/posts/fwer.html) one common method for dealing with testing multiple simultaneous hypotheses, methods that control the Family-wise error rate. However, we realized that the FWER can be quite conservative. The False-Discovery rate is a powerful alternative to the FWER, which is often used in cases where hundreds or thousands of simultaneous hypotheses are tested. What is the FDR, and how is it different from the FWER? How do we control it? What price do we pay for doing so*

# What happens to the FWER when we test hundreds or thousands of hypotheses?

# The goal of FDR control: Make sure few of your findings are spurious

https://web.stanford.edu/~hastie/CASI_files/PDF/casi.pdf, ch.15, epage 289

That matrix that shows up everywhere

# FDR control when the hypotheses are independent: Benjamini-Hochberg

# FDR control under arbitrary dependence: Benjamini–Yekutieli

# An example with independence

https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html

# An example with dependence: Pairwise comparisons

https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html

https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html
