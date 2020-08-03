---
layout: post
title: "Conjugate priors for normal (and not-so-normal) data: The Normal-Gamma model"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: jellybeans.png
---

Bayesian analysis of proportions using the beta conjugate prior is relatively easy to automate once you've got the basics down ([this](https://www.quantstart.com/articles/Bayesian-Inference-of-a-Binomial-Proportion-The-Analytical-Approach/), [this](https://stephens999.github.io/fiveMinuteStats/bayes_beta_binomial.html), and [this](https://www.youtube.com/watch?v=D0CjtN8RYWc) are some good references if it's a topic that's new to you or you'd like a refresher). For practitioners using Python, it's not much harder than `from scipy.stats import beta`. Analyzing the mean of continuous data is a little more slippery, though. The compound distribution named in the [Wikipedia page](https://en.wikipedia.org/wiki/Conjugate_prior#When_likelihood_function_is_a_continuous_distribution), the frightening-sounding "Normal Gamma", isn't even implemented in scipy. What's an analyst who just wants to evaluate a posterior distribution of the mean and variance to do?

Well, it turns out it's not too bad - there's just some assembly required. It's worth a tour of the theory to get the context, and then we'll do an example in Python.

# A running example: Changes in clicks per customer

It's often helpful to think about analysis techniques in terms of an example. We'll take a look at an actual set of data to get a feel for how we might apply a normal conjugate prior in practice to understand the mean and variance of a distribution.

Let me set the scene. You're the CBO (Chief Bayes Officer) of paweso.me, purveyor of deep learning blockchain AI targeted advertising for cats. You recently worked with some of your engineers to launch ChonkNet™, a Neural Network which predicts which users will buy which products (you're hoping it's an improvement on your current targeting model, DeepMeow™). You show a few thousand random users products based on the new algorithm, and measure the number of ads they clicks this week compared to compared to last week. The resulting per-user `this week clicks - last week clicks` looks like this:<sup>[1](#foot1)</sup> 

![clicks per user histogram](https://github.com/lmc2179/lmc2179.github.io/blob/master/assets/img/bayes_norm/clicksperuser.png)

You'd like to know: What does the posterior of the mean and variance of revenue-per-user look like?

# The usual story: Standard errors and confidence intervals

A quick refresher: In the classical story, quantities like "the population mean" and "the population variance" are actual properties of the population floating around in the world. We assume that the data we have were sampled from the population according to some random process, and then we make some inferences about what the population quantities might be. Usually, we can't recover them exactly, but we can come up with a reasonable guess (a point estimate) and a range of variation based on the way the sampling happened (the confidence interval). If we like, we can even compute a statistic to gauge whether or not we can reject the hypothesis that our favorite population quantity is exactly zero, or test some other hypothesis we're interested in (compute a P-value).

The kindest version of this story is one where the quantity we want is the population mean, the sampling is totally random, and we can appeal trust the Central Limit Theorem to tell us the sampling distribution of the mean. In that case, the sampling distribution of the mean is centered around the (unknown) true mean, and the standard deviation of the sampling distribution is the [standard error](https://en.wikipedia.org/wiki/Standard_error#Standard_error_of_the_mean).

The standard error of the mean is such an important formula that I'll note it here. It will also turn out that the classical standard error has an important relationship with the Bayesian story we're about to tell. If the mean of a random variable is $\mu$, the standard deviation is $\sigma$, and the sample size is $n$, then the standard error is

$$SE_\mu = \frac{\sigma}{\sqrt{n}}$$

In practice, we often don't know the population standard deviation, $\sigma$. When the sample is "large", we're usually willing to use a point estimate of sigma computed from the data. This sweeps a bit of uncertainty associated with sigma under the rug - if we wanted to avoid doing so, we'd use a [slightly different procedure based on the T-distribution](https://en.wikipedia.org/wiki/Standard_error#Student_approximation_when_%CF%83_value_is_unknown). 

The standard error is so valuable in part because it lets us construct a confidence interval around the sample mean. The idea is that if we always constructed the CI at level $\alpha$ around the sample mean, our interval will contain the true mean most of the time. Specifically, we'd only leave it out $\alpha$ proportion of the time.

In the case of our datast from above, we might produce something like this:

```python
from scipy.stats import norm

se = np.std(changes_in_clicks ) / np.sqrt(len(changes_in_clicks ))

print('The 99% confidence interval: {0}'.format(norm.interval(0.99, loc=np.mean(changes_in_clicks ), scale=se)))
```

Output:
```
The 99% confidence interval: (3.8489829444627057, 4.2390170555372935)
```

So the classical analysis would tell us that the week-over-week change in clicks per user is between about 3.84 and 4.24 at the 99% level.

# A Refresher: Priors, Posteriors, Likelihoods, and Bayesian updates

The Bayesian perspective is a little different. Like the classical picture, it involves two parameters of interest - the mean and the variance, commonly referred to as $\mu$ and $\sigma^2$.. We'd like to learn something about these values of these parameters from the data. The Bayesian procedure involves a few steps:
- Write down your prior, $\mathbb{P}(\mu, \sigma)$, which summarizes all the information you have about $\mu$ and $\sigma$ before you see the data. This might be as vague as "every value of these parameters is equally likely" or as specific as "I'm almost entirely sure that $\mu$ is between 2 and 4".
- Pick a Likelihood function $\mathcal{L}(X_1, ..., X_n | \mu, \sigma)$ that relates the data to a potential value of the parameters.
- Look at the observed data, the values of $X$ that actually showed up in our dataset.
- Use Bayes Theorem to update the prior and get the posterior, $\mathbb{P}(\mu, \sigma | X_1, ..., X_n)$. 

We can look at the posterior to learn everything we want to know

# The Bayesian version

In the both the Bayesian and classical version of this story, we're interested in learning about the mean. In the last section, we compute a point estimate of the mean, and then come up with the standard error to construct confidence intervals around that estimate. From the Bayesian perspective, we choose a prior distribution, then combine it with the data to get a posterior distribution that tells us about our knowledge of the mean's value. 

The main difference between these views is that in the classical view, all our knowledge comes from the sampling distribution and what the data tells us about it. In the Bayesian view, all our knowledge comes from the posterior distribution we calculate from the prior and the data. Endnote: It's not worth discusssing the philosophical differences between these two perspectives in detail right now. There are real advantages and disadvantages to the tools developed by both sides to be discussed another time, but [the idea of a century-long holy war between Bayesians and Frequentists is overrated when we can all be pragmatists.](https://arxiv.org/pdf/1106.2895v2.pdf)

Let's assume we know the variance
(We often don't know the true variance. a quick fix is to plug in the estimated variance and act as if that's the real variance. if that makes you nervous, don't worry - we'll explore the implications of it a little later

Wiki says:

$$\mu | \mu_0, \sigma_0, \sigma, n \sim N(\mu_{post}, \sigma_{post}) $$

$$\mu_{post} = \frac{1}{\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2}} \left ( \frac{\mu_0}{\sigma_0^2} + \frac{\sum_i x_i}{\sigma^2} \right )$$

$$\sigma_{post}^2 = \left (\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2} \right) ^{-1}$$

woah now. let's break that down - how does changing these parameters affect our inference


Sometimes, we say we don't have any prior information, and so we pick a "flat" prior (I'll refrain from calling it uninformative)

```python
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import norm

n = 100
true_mu = 5
true_sd = 2

rng = np.random.default_rng()

x = rng.normal(true_mu, true_sd, n)

sample_sd = np.std(x, ddof=1)
sample_mean = np.mean(x)
estimated_se = sample_sd /  np.sqrt(len(x))

posterior_mean_samples = rng.normal(sample_mean, estimated_se, 1000)

sns.distplot(posterior_mean_samples)
plt.axvline(sample_mean, label='Sample mean')
plt.axvline(true_mu, label='True mean', color='orange')
lower, upper = np.quantile(posterior_mean_samples, (.01, .99))
plt.scatter([lower, upper], [0, 0], marker='s', color='green', s=100, label='CI bounds')
plt.legend()
plt.show()
```

# The posterior of the mean is closely connected to the standard error of the mean

f

$$\lim_{\mu_0 \rightarrow 0, \sigma_0 \rightarrow \infty}\mu_{post}  = 
\frac{1}{0 + \frac{n}{\sigma^2}} \left ( 0 + \frac{\sum_i x_i}{\sigma^2} \right ) =\frac{\frac{\sum_i x_i}{\sigma^2}}{\frac{n}{\sigma^2}} = \frac{\sum_i x_i}{n}$$

$$\sigma_{post}^2 = \left (\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2} \right) ^{-1}$$

$$\lim_{\mu_0 \rightarrow 0, \sigma_0 \rightarrow \infty} \sigma_{post}^2 
= \left (\frac{1}{\sigma_0^2} + \frac{n}{\sigma^2} \right) ^{-1}
= \frac{\sigma^2}{n} = \sqrt{SE_\mu}$$

# But I _don't_ know the variance!!

Okay, okay! Let's see how we might incorporate the uncertainty about the variance into our analysis, and how that changes our analysis if we mainly care about the mean - if the variance is a "nuisance parameter", as it often is.

If we don't: Normal Gamma

https://richrelevance.com/blog/2013/07/31/bayesian-analysis-of-normal-distributions-with-python/

$$\mu, \tau \sim N \Gamma (\mu_0, \nu, \alpha, \beta) $$

We observe $x$

|Parameter|Updated value|Flat prior value|Posterior value when prior is flat|
|---|---|---|---|
|$\mu_0$|$\frac{\nu \mu_0 + n \bar{x}}{\nu + n}$|$0$|$\bar{x}$|
|$\nu$|$\nu + n$|$0$|$n$|
|$\alpha$|$\alpha + \frac{n}{2}$|$-\frac{1}{2}$|$\frac{n}{2} - \frac{1}{2}$|
|$\beta$|$$|||

Unlike in the known-variance case, this does not really produce

https://github.com/urigoren/conjugate_prior/blob/master/conjugate_prior/invgamma.py

```python
from scipy.stats import gamma
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

a = -0.5 # This recovers the bessel-corrected MLE; maybe that's an endnote though since it's not a real prior (but I won't tell if you won't)
b = 0

x = np.random.normal(0, 1, 10)
a_posterior = a + len(x) / 2
b_posterior = b + (0.5 * np.sum((x - np.mean(x))**2))
prec_posterior = gamma(a_posterior, scale=1./b_posterior)
precision_samples = prec_posterior.rvs(1000)
var_samples = 1./precision_samples
sns.distplot(var_samples)
plt.axvline(np.var(x, ddof=1))
plt.axvline(1./prec_posterior.mean(), color='orange', linestyle='dotted')
plt.show()
```

```python
# An intuitive justification for the Gamma(0, 0) prior - Variance prior is arbitrarily broad
eps_choices = np.linspace(1e1, 1e-6, 500)
lower = np.array([gamma(eps, scale=1./eps).interval(.99)[0] for eps in eps_choices])
upper = np.array([gamma(eps, scale=1./eps).interval(.99)[1] for eps in eps_choices])
plt.plot(eps_choices, 1./lower, label='Upper')
plt.plot(eps_choices, 1./upper, label='Lower')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.title('Upper and lower bounds on prior variance as epsilon changes')
plt.show()
```

Endnote stuff: (1) This isn't a real prior so what is it (2) Bessel correction (3) This illustrates from the Bayesian point of view why the normal distribution isn't a great posterior when the sample is small but the T-distribution is; the normal distribution doesn't include the uncertainty from the unknown variance, leading to thinner tails.

https://stackoverflow.com/questions/42150965/how-to-plot-gamma-distribution-with-alpha-and-beta-parameters-in-python

https://en.wikipedia.org/wiki/Normal-gamma_distribution#Generating_normal-gamma_random_variates

# How should I pick my prior?

For the Bayes-skeptical

Those of you who have not yet accepted Reverend Thomas Bayes into your heart may

v = 0, mean = 0, alpha=0, beta=0

When we use the above prior it corresponds to MLE and has good frequentist properties

# How essential is it that my data look normal

I mean what are a few outliers among friends eh

This shows up when the data are averages

Simulation of an almost-normal example

Large sample size usually makes up for non-normality; https://dspace.mit.edu/bitstream/handle/1721.1/45587/18-441Spring-2002/NR/rdonlyres/Mathematics/18-441Statistical-InferenceSpring2002/C4505E54-35C3-420B-B754-75D763B8A60D/0/feb192002.pdf

compare with Bayesian bootstrap

https://scholarsarchive.byu.edu/cgi/viewcontent.cgi?article=1277&context=facpub

# An extra tip: What if my data is Log Normal instead of Normal?

Here's a neat trick.

# Endnotes

<a name="foot1">1</a> Okay, fine, that's not real data; ChonkNet™ remains just an unrealized daydream in the head of one data scientist. I actually generated this data with this code: 

```python
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm, expon

np.random.seed(7312020)

changes_in_clicks = skellam(5, 1).rvs(1000)

sns.distplot(changes_in_clicks, kde=False, bins=np.arange(-5, 15))
plt.xlabel('Observed change in clicks per user')
plt.show()
```

The idea is that the number of clicks is Poisson distributed in each week, and the [Skellam distribution](https://en.wikipedia.org/wiki/Skellam_distribution) is the distribution of the difference between two Poisson distributions. So this is hopefully a plausible set of data that we might see from the difference of two weekly click datasets.
