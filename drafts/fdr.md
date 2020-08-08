---
layout: post
title: "The False-Discovery Rate: An alternative to the FWER for multiple comparisons"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: jellybeans.png
---

*We've [previously explored](https://lmc2179.github.io/posts/fwer.html) one common method for dealing with testing multiple simultaneous hypotheses, methods that control the Family-wise error rate. However, we realized that the FWER can be quite conservative. The False-Discovery rate is a powerful alternative to the FWER, which is often used in cases where hundreds or thousands of simultaneous hypotheses are tested. What is the FDR, and how is it different from the FWER? How do we control it? What price do we pay for doing so*

# What happens to the FWER when we test hundreds or thousands of hypotheses?

In our discussion of the FWER, we walked through some strategies for avoiding Type I errors when we test multiple simulataneous hypotheses. It turned out that when we tested 5 hypotheses instead of one, we might accidentally reject a true null hypothesis much more often than $\alpha$, our significant level. We looked at the Bonferroni and Bonferroni-Holm methods, which let us ensure that the probabilty of any false claims at all was less than $\alpha$.

Let's imagine a scenario where instead of testing 5 hypotheses, we're testing 5000. While this might seem a little far fetched, it occurs pretty frequently:
- Machine learning models might have hundreds or thousands of features which we'd like to test for their correlation with the output
- [Microarray](https://en.wikipedia.org/wiki/DNA_microarray) studies involve looking at the expression of thousands of genes
- Experiments might cast a wide net and screen many possible treatments for potential value

In a case like this, our analysis might produce a giant number of significant results. An experiment like this might involve rejecting hundreds of null hypotheses. If that's the case, the FWER is pretty strict - it will ensure that we very rarely make even one false statement. But in a lot of these cases, we're intentionally casting a wide net, and we don't need to be so conservative. We're often perfectly happy to reject 250 null hypotheses when we should have only rejected 248 of them; we still found 148 new and exciting relationships we can explore! The FWER-controlling methods, though, will work hard to make sure this doesn't happen.

# The goal of FDR control: Make sure few of your findings are spurious

The FWER, it turns out, is just one way of thinking about the Type I error rate when we test multiple hypotheses. In the case above, we had two false positives; but we had so many true positives that it wasn't an especially big deal.

For every hypothesis, there are four outcomes:
- False Positive
- True Positive
- False Negative
- True Negative

![Matrix](https://raw.githubusercontent.com/lmc2179/lmc2179.github.io/master/assets/img/fdr/fdr_matrix.png)

- There are $a$ False positives, $b$ True positives,

https://web.stanford.edu/~hastie/CASI_files/PDF/casi.pdf, ch.15, epage 289

We can use this matrix to summarize the FWER and FDR:
- FWER-controlling methods attempt to keep $\frac{a}{N_0 + N_1} \leq \alpha$
- FDR-controlling methods attempt to keep the average $\frac{a}{a + b}$ at $\alpha$.

# FDR control when the hypotheses are independent: Benjamini-Hochberg

http://www.math.tau.ac.il/~ybenja/MyPapers/benjamini_hochberg1995.pdf

# Alternatives to the Benjamini-Hochberg procedure

Benjamini–Yekutieli
no independence assumption
However the power is lower
http://www.math.tau.ac.il/~ybenja/MyPapers/benjamini_yekutieli_ANNSTAT2001.pdf

A close relative of the FDR is the False coverage rate 
https://en.wikipedia.org/wiki/False_coverage_rate

# An example: Feature selection in a linear model

```python
from sklearn.datasets import make_regression
from statsmodels.api import OLS
import numpy as np
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns

def simulate_fdr():
  n_tests = 1000
  n_sample = 2000
  n_true_pos = 100

  alpha = .05

  X, y, coef = make_regression(n_samples=n_sample, n_features=n_tests, n_informative=n_true_pos, bias=0, coef=True, noise=200)
  p = OLS(y, X).fit().pvalues
  reject = p <= alpha
  true_null = (coef == 0)
  count_discoveries = np.sum(reject)
  count_false_discoveries = np.sum(reject & true_null)

  reject_bh = multipletests(p, method='fdr_bh', alpha=alpha)[0]
  count_discoveries_bh = np.sum(reject_bh)
  count_false_discoveries_bh = np.sum(reject_bh & true_null)

  naive_fdr = count_false_discoveries / count_discoveries
  bh_fdr = count_false_discoveries_bh / count_discoveries_bh
  return naive_fdr, bh_fdr
  
simulations = [simulate_fdr() for _ in tqdm(range(1000))]

naive_fdr_dist, bh_fdr_dist = zip(*simulations)

sns.distplot(naive_fdr_dist, label='Naive method')
sns.distplot(bh_fdr_dist, label='BH method')
plt.title('FDR distribution from 1000 simulations')
plt.axvline(.05, linestyle='dotted')
plt.axvline(np.mean(bh_fdr_dist), label='Observed FDR for BH', color='green')
plt.legend()
plt.show()
```

![FDR simulation results](https://raw.githubusercontent.com/lmc2179/lmc2179.github.io/master/assets/img/fdr/fdr_regression.png)

```python
sorted_p, sorted_true_null = zip(*sorted(zip(p, true_null)))
sorted_p = np.array(sorted_p)
sorted_true_discoveries = (1. - np.array(sorted_true_null)).astype(np.bool)

alpha = .05
i = np.arange(len(sorted_p)) + 1
m = len(sorted_p)
holm_cutoffs = alpha / (m + 1 - i)
hochberg_cutoffs = (alpha * i) / len(sorted_p)
plt.plot(sorted_p, label='P-values, sorted')
plt.plot(holm_cutoffs, label='Bonferroni-Holm')
plt.plot(hochberg_cutoffs, label='Benjamini-Hochberg')
plt.scatter(i[sorted_true_discoveries]-1, sorted_p[sorted_true_discoveries], marker='x')
plt.legend()
plt.show()
```

# What else might we do?

Estimating multiple values simultaneously is a James-Stein/Shrinkage/Hierarchical analysis problem; rather than looking at the long-run null rates, look at the quality of the estimate
