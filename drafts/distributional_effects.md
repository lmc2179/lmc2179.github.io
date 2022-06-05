---
layout: post
title: "How did my treatment affect the distribution of my outcomes? A/B testing with quantiles and their confidence intervals in Python"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: distributional_effects.png
---

*?*

*[Photo credit](https://pxhere.com/en/photo/932166)*


# Distributional effects of A/B tests are often overlooked but provide a deeper understanding

## The group averages and average treatment effect hide a lot of information

Most companies I know of that include A/B testing in their product development process usually do something like the following for most of their tests:
* Pick your favorite metric which you want to increase, and perhaps some other metrics that will act as guard rails. Often, this is some variant of "revenue per user", "engagment per user", ROI or the efficiency of the process.
* Design and launch an experiment which compares the existing product's performance to that of some variant products.
* At some point, decide to stop collecting data.
* Compute the average treatment effect for the control version vs the test variant(s) on each metric. Make a decision about whether to replace the existing production product with one of the test variants.

This process is so common because, well, it works - if followed, it will usually result in the introduction of product features which increase our favorite metric. It is a series of discrete steps in the product space which attempt to optimize the favorite metric without incurring unacceptable losses on the other metrics.

In this process, the average treatment effect is the star of the show. But as we learn in Stats 101, two distributions can look drastically different while still having the same average. For example, here are four remarkably different distributions with the same average:

```python
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import poisson, skellam, nbinom, randint, geom

for dist in [poisson(100), skellam(1101, 1000), randint(0, 200), geom(1./100)]:
  plt.plot(np.arange(0, 400), dist.pmf(np.arange(0, 400)))
plt.xlim(0, 400)  
plt.ylabel('PMF')
plt.title('Four distributions with a mean of 100')
plt.show()
```

![Four distributions with different shapes but the same mean](https://raw.githubusercontent.com/lmc2179/lmc2179.github.io/master/assets/img/distributional_effects/Figure_0.png)

Similarly, the average treatment effect does not tell us much about how our treatment changed the shape of the distribution of outcomes. But we can expand our thinking not just to consider how the treatment changed the average, but the effect on the shape of the distribution; the [distributional effect](https://en.wikipedia.org/wiki/Distributional_effects) of the treatment. Expanding our thought to think about distributional effects might give us insights that we can't get from averages alone, and help us see more clearly what our treatment did. For example:

* If we have a positive treatment effect, we can see whether one tail of the distribution was disproportionately affected. Did our gains come from lifting everyone? From squeezing more revenue out of the high-revenue users? From "lifting the floor" on the users who aren't producing much in control?
* If an experiment negatively affected one tail of the distribution, we can consider mitigation. If our treatment provided a negative experience for users on the low end of the distribution, is there anything we can do to make their experience better?
* Are we meeting our goals for the shape of the distribution? For example, if we want to maintain a minimum service level, are we doing so in the treatment group?
* Do we want to move up market? If so, is our treatment increasing the output for the high end of the outcome distribution?
* Do we want to diversify our customer base? If so, is our treatment increasing our concentration among already high-value users?

The usual average treatment effect cannot answer these questions. We could compare single digit summaries of shape (variance, skewness, kurtosis) between treatment and control. However, even these are only simplified summaries; they describe a single attribute of the shape like the dispersion, symmetry, or heavy tailedness.

Instead, we'll look at the empirical [quantiles](https://en.wikipedia.org/wiki/Quantile) of control and treatment, and the difference between them. We'll lay out some basic definitions here:
* We'll call
* Empirical quantile
* Quantile curve
* the inverse of the quantile curve is the CDF , and its empirical counterpart is the [empirical CDF](https://www.statsmodels.org/devel/generated/statsmodels.distributions.empirical_distribution.ECDF.html)

Let's take a look at an example of how we might use these in practice to learn about the distributional effects of a test.

# An example: How did my A/B test affect

Let's once more put ourselves in the shoes of that most beloved of Capitalist Heroes, the [purveyor of little tiny cat sunglasses](https://lmc2179.github.io/posts/confidence_prediction.html). Having harnessed the illuminating insights of your business' data, you've consistently been improving your key metric of Revenue per Cat. You currently send out a weekly email about the current purrmotional sales, a newsletter beloved by dashing calicos and tabbies the world over. As you are the sort of practical, industrious person who is willing to spend their valuable time reading a blog about statistics, you originally gave this email the very efficient subject line of "Weekly Newsletter" and move on to other things. 

However, you're realizing it's time to revisit that decision - your previous analysis demonstrated that warm eather is correlated with stronger sales, as cats everywhere flock to sunny patches of light on the rug in the living room. Perhaps, if you could write a suitably eye-catching subject line, you could make the most of this seasonal oppourtunity. Cats are notoriously aloof, so you settle on the overstuffed subject line "**W**ow so chic ✨ shades 🕶 for cats 😻 summer SALE ☀ _buy now_" in a desperate bid for their attention. As you are (likely) a person and not a cat, you decide to run an A/B test on this subject line to see if your audience likes the new subject line.

You fire up your A/B testing platform, and get 1000 lucky cats to try the new subject line, and 1000 to try the old one. You measure the revenue purr customer in the period after the test, and you're ready to analyze the test results.

Lets import some things from the usual suspects:

```python
from scipy.stats import norm, sem # Normal distribution, Standard error of the mean
from copy import deepcopy 
import pandas as pd
from tqdm import tqdm # A nice little progress bar
from scipy.stats.mstats import mjci # Calculates the standard error of the quantiles: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.mquantiles_cimj.html
from matplotlib import pyplot as plt # Pretty pictures
import seaborn as sns # Matplotlib's best friend
import numpy as np 
```

Histogram

```python
plt.title('Distribution of revenue per customer')
sns.distplot(data_control, label='Control')
sns.distplot(data_treatment, label='Treatment')
plt.ylabel('Density')
plt.xlabel('Revenue ($)')
plt.legend()
plt.show()
```

![Histogram](https://raw.githubusercontent.com/lmc2179/lmc2179.github.io/master/assets/img/distributional_effects/Figure_1.png)

Hm. That's a little tough to read. Just eyeballing it, the tail on the Treatment group seems a little thicker, but it's hard to say much more than that.

The usual estimate of treatment effect

```python
def z_a_over_2(alpha):
  return norm(0, 1).ppf(1.-alpha/2.)

te = np.mean(data_treatment) - np.mean(data_control)
ci_radius = z_a_over_2(.05) * np.sqrt(sem(data_treatment)**2 + sem(data_control)**2)
print('Average treatment effect: ', te, '+-', ci_radius)
```

```
Average treatment effect:  1.1241231969779277 +- 0.29768161367254564
```

Now lets think distributionally;
* Is the gain coming from squeezing more out of the big spenders or increasing engagement with those who spend least
* Was any part of the distribution negatively affected

Box and whisker - uh, hm

```python
Q = np.linspace(0.05, .95, 20)

plt.boxplot(data_control, positions=[0], whis=[0, 100])
plt.boxplot(data_treatment, positions=[1], whis=[0, 100])
plt.xticks([0, 1], ['Control', 'Treatment'])
plt.ylabel('Revenue ($)')
plt.title('Box and Whisker - Revenue per customer by Treatment status')
plt.show()
```

![Box and Whisker plot](https://raw.githubusercontent.com/lmc2179/lmc2179.github.io/master/assets/img/distributional_effects/Figure_2.png)

Quantiles - There's something a little clearer

```python
plt.title('Quantiles of revenue per customer')
plt.xlabel('Quantile')
plt.ylabel('Revenue ($)')
control_quantiles = np.quantile(data_control, Q)
treatment_quantiles = np.quantile(data_treatment, Q)
plt.plot(Q, control_quantiles, label='Control')
plt.plot(Q, treatment_quantiles, label='Treatment')
plt.legend()
plt.show()
```

![Quantile plots](https://raw.githubusercontent.com/lmc2179/lmc2179.github.io/master/assets/img/distributional_effects/Figure_3.png)

Quantile difference

With MJ https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.mjci.html

```python
plt.title('Quantile difference (Treatment - Control)')
plt.xlabel('Quantile')
plt.ylabel('Treatment - Control')
quantile_diff = treatment_quantiles - control_quantiles
control_se = mjci(data_control, Q)
treatment_se = mjci(data_treatment, Q)
diff_se = np.sqrt(control_se**2 + treatment_se**2)
diff_lower = quantile_diff - z_a_over_2(.05 / len(Q)) * diff_se
diff_upper = quantile_diff + z_a_over_2(.05 / len(Q)) * diff_se
plt.plot(Q, quantile_diff, color='orange')
plt.fill_between(Q, diff_lower, diff_upper, alpha=.5)
plt.axhline(0, linestyle='dashed', color='grey', alpha=.5)
plt.show()
```

![Quantile difference](https://raw.githubusercontent.com/lmc2179/lmc2179.github.io/master/assets/img/distributional_effects/Figure_4.png)

Big takeaways; effect in upper quartiler; the fat cats get fatter; feline fashionistas buy up all the sale inventory

# Outro: Other ideas and alternatives

* Hetereogeneous effect analysis/subgroup analysis: Possible introduction of mitigation strategy
* Conditional variance modeling: Also a way of understanding change in the shape (conditional kurtosis? don't think anyone ever does that)
* Change in Gini, change in Entropy, change in https://en.wikipedia.org/wiki/Income_inequality_metrics#Gini_index
* Many variables: Quantile regression is a good framework

# DGP

```python
sample_size = 1000
data_control = np.random.normal(0, 1, sample_size)**2
data_treatment = np.concatenate([np.random.normal(0, 0.01, round(sample_size/2)), np.random.normal(0, 2, round(sample_size/2))])**2
```