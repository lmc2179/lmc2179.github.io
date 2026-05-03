# Error propagation with the Delta Method

Possibly https://www.explainxkcd.com/wiki/images/9/9c/error_bars.png

_Any time we use data to answer a question, we need to be mindful of our error bars. That means not just calculating standard errors on our estimates, but propagating those standard errors when we do other calculations with them. The Delta Method is a way of propagating error if your transformation has a well-defined equation. This guide will walk you through the key ideas of the Delta Method and how to use it in practice._

> Indeed, it is likely that we will always be mired in error. The most each generation can hope for is to reduce the error bars a little, and to add to the body of data to which error bars are attached. The error bar is a pervasive, visible self-assessment of the reliability of our knowledge. You can often see error bars in public opinion polls ("an uncertainty of plus or minus 3 percent," say). It would be a step forward were error bars or their equivalent prominently displayed in politics, economics, and religion.

~Carl Sagan in  _The Demon-Haunted World_, earning his spot in my dream blunt rotation

# Error propagation is everywhere

Any time you estimate something using data, your audience will appreciate it if you include error bars. Error bars (confidence intervals, standard errors, credible intervals, etc) help your audience understand how seriously to take the estimate. Sometimes we are lucky, and we know a lot, and the error bars are very narrow. More commonly, the error bars are wider than we would like - and so it is extra important to track them and communicate them.

Sometimes you use a measured quantity to calculate something else, and you need to propagate the error through your calculation, so you'll know the error bars on the final number.

An example: data collection example with an isoelastic relationship; A/B test measures effect on ? which affects ? isoelastically. Maybe like: AI eval vs experience quality

One option - use the bootstrap, or jackknife. Delta method relies on more assumptions but is much faster than the bootstrap. But a closed form is easiest to work with, both for speed and intuition

# How the delta method propagates error

Let's step through the error propagation story again more formally

Little sapling data scientists, coming up in the Garden of Analysis, learn early on that their best friend is the **sample mean**:

$$\hat{\mu} = \frac{1}{n} \sum_{i=1}^{i=n} y_i$$

They also learn that statements about the sample mean are most useful if you calculate the **standard error of the sample mean**:

$$\hat{SE}(\hat{\mu})= \frac{\sigma}{\sqrt{n}}$$
m
This calculation is at the heart of many practical applications, especially, clinical trials and A/B tests. It's a powerful tool, letting practitioners make inferences about populations in a very wide variety of situations. Armed with the sample mean, we can compute confidence intervals, hypothesis tests, and lots of other practically useful things.

Let's start with the univariate case. If $f(\theta)$ is our output, then

$\hat{SE}(f(\hat{\theta})) \approx \mid f^\prime (\hat{\theta}) \mid \times \hat{SE}(\hat{\theta}) $

And so an approximate asymptotic confidence interva for $f(\hat{\theta})$ is 

$f(\hat{\theta}) \pm z_{\alpha/2} \times \mid f^\prime (\hat{\theta}) \mid \times \hat{SE}(\hat{\theta})$

Under this approximation, note that
* The CI is larger for transformations that vary a lot at $\hat{\theta}$ due to the reliance on the first derivative
* The CI is smaller when the SE of $\hat{\theta}$ is smaller

The delta method tells us how much the function "inflates" the the SE of $\hat{\theta}$; it inflates it by $\mid f^\prime (\hat{\theta}) \mid$

Keep in mind that this is asymptotic. You can compare w/ bootstrap to verify

# A univariate example with a comparison to the bootstrap

# The multivariate version of the story

Note that in this section $f: \mathbb{R}^k \rightarrow \mathbb{R}$

$\hat{SE}[f(\hat{\theta})] \approx \sqrt{\nabla f (\hat{\theta})^T Cov(\hat{\theta}) \nabla f (\hat{\theta})}$

Dimensions: (1) = (1 x k) (k x k) (k x 1)

Recall that $\nabla f (\hat{\theta})$ is the kx1 column vector of partial derivatives

When the elements of $\hat{\theta}$ are uncorrelated, then $Cov(\hat{\theta}) = I (\sigma_1^2, ..., \sigma_k^2)$, ie just the identity matrix with the main diagonals as the sd

and

$\hat{SE}[f(\hat{\theta})] = \sqrt{\sum_i \frac{df}{d\hat{\theta}}(\hat{\theta})^2 \times \sigma_i^2}$ [Uncorrelated case]

Uncorrelated example: Rev from multiple inputs added together

Challenge: Correlated RVs work but you need to know the correlation

# A multivariate example without a simulation

? May not include

# Recap: A step-by-step guide to using the delta method

1. Figure out your measured quantity and derived quantity.
    1. The derived quantity is some known function of the measured quantity.
2. Find the standard error of the measured quantity.
3. Propagate through the standard error using the delta method. (requires derivative, link sympy post)

# appendix: some worked examples

## Difference of means (uncorrelated)

$\hat{\Delta} = \hat{\mu_2} - \hat{\mu_1}$

$\hat{SE}(\hat{\theta}) = \sqrt{1 \times \hat{SE}(\hat{\mu_2})^2 + (-1) \times \hat{SE}(\hat{\mu_1})^2} = \sqrt{\hat{SE}(\hat{\mu_2})^2 - \hat{SE}(\hat{\mu_1})^2}$

## Log of means

$\hat{l} = log(\hat{\mu}); \frac{d \hat{l}}{d \hat{\mu}} = \frac{1}{\hat{\mu}}$

## Mean plus a constant

$\hat{\delta} = \alpha + \hat{\mu}$

$\hat{SE}(\hat{\delta}) = \hat{SE}(\hat{\mu})$

## Mean times a constant

$\hat{\delta} = \alpha \times \hat{\mu}$

$\hat{SE}(\hat{\delta}) = \alpha \times \hat{SE}(\hat{\mu})$

# appendix: intuition about how it works

Intuition about the key theorem from Shalizi

Normal assumptions?

1. Taylor series expand f (1st order/linear) around $\theta^*$, the true value. C.2 approximate the value at the MLE, C.3, C.4.
2. Use rules for linear combination of variance to get C.8

Solve nonlinear combinations of variance by linearizing them with the taylor series

All of statistics section 5.5, Thm 5.13