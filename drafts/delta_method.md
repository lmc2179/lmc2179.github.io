
A step-by-step guide to propagating error using the Delta Method

# Error Propagation is Everywhere

The whole point of statistical analysis is to quantify uncertainty. The results of even very complex analysis can often be summed up by a statement like: 

> Under certain assumptions which seem plausible, the data indicates that a reasonable estimate for the random variable $\theta$ is $\hat{\theta}$, with a standard error of $\hat{SE}(\hat{\theta})$.

We have various methods of extracting an estimate plus a standard error from the data - we know the standard error of the mean, we can use OLS, etc. We can then propagate error using error propagation rules (link). But what if your case is not covered by those rules? We can imagine some examples:

* Lift, or ratios of two random variables $f(\theta_1, \theta_2) = \frac{\theta_2}{\theta_1}$
* Multiply by a constant (forex, yield) $f(\theta) = \alpha \theta$, where $\alpha$ is the exchange rate
* Retention <-> Churn (1/100 pct churn is 100 expected retention). $f(\theta) = \theta ^{-1}$
* We estimated an increase of \Delta in the input; how does it translate to a delta on the output if we know the production function. $f(\theta) = \alpha \theta^\beta$

These are hard because we know the SE of inputs, but not the outputs. If you transformation has a nice form (and a 1st derivative), the delta method lets you propagate the error from the ones you know, to the ones you don't.

# How it works - Univariate

Each of the scenarios outlines has _a set of statistics with known standard errors_ (we'll call them $\theta$) as well as _a set of statistics with unknown standard errors_ (we'll call them $f(\theta)$). $f$ is the function mapping the two sets. For example ...

Let's start with the univariate case. If $f(\theta)$ is our output, then

$\hat{SE}(f(\hat{\theta})) \approx \mid f^\prime (\hat{\theta}) \mid \times \hat{SE}(\hat{\theta}) $

And so an approximate asymptotic confidence interva for $f(\hat{\theta})$ is 

$f(\hat{\theta}) \pm z_{\alpha/2} \times \mid f^\prime (\hat{\theta}) \mid \times \hat{SE}(\hat{\theta})$

Under this approximation, note that
* The CI is larger for transformations that vary a lot at $\hat{\theta}$ due to the reliance on the first derivative
* The CI is smaller when the SE of $\hat{\theta}$ is smaller

The delta method tells us how much the function "inflates" the the SE of $\hat{\theta}$; it inflates it by $\mid f^\prime (\hat{\theta}) \mid$

Keep in mind that this is asymptotic. You can compare w/ bootstrap to verify

All of statistics section 5.5, Thm 5.13

# Univariate example

See the appendix for worked examples that we "already know"

## Isoelastic example

You estimate that the input is $\hat{\theta}$; how does this propagate to the output?

$\frac{d}{dx} \alpha x^\beta = \beta \alpha x^{\beta - 1}$

So the $SE$ is 

$(\beta \alpha \hat{\theta}^{\beta - 1}) \times SE(\hat{\theta}) = SE(y)$

Derive

$f(\theta)$

Calc

Compare bootstrap

# How it works - Multivariate

Note that in this section $f: \mathbb{R}^k \rightarrow \mathbb{R}$

$\hat{SE}[f(\hat{\theta})] \approx \sqrt{\nabla f (\hat{\theta})^T Cov(\hat{\theta}) \nabla f (\hat{\theta})}$

Dimensions: (1) = (1 x k) (k x k) (k x 1)

Recall that $\nabla f (\hat{\theta})$ is the kx1 column vector of partial derivatives

When the elements of $\hat{\theta}$ are uncorrelated, then $Cov(\hat{\theta}) = I (\sigma_1^2, ..., \sigma_k^2)$, ie just the identity matrix with the main diagonals as the sd

and

$\hat{SE}[f(\hat{\theta})] = \sqrt{\sum_i \frac{df}{d\hat{\theta}}(\hat{\theta})^2 \times \sigma_i^2}$ [Uncorrelated case]

Uncorrelated example: Rev from multiple inputs added together

Challenge: Correlated RVs work but you need to know the correlation

# Multivariate example

Sum of isoelastic functions

# Intuition about the key theorem from Shalizi

Normal assumptions?

1. Taylor series expand f (1st order/linear) around $\theta^*$, the true value. C.2, C.3, C.4.
2. Use rules for linear combination of variance to get C.8

Solve nonlinear combinations of variance by linearizing them with the taylor series

# Other ideas: Alternatives to the delta method

Jackknife/bootstrap

Transform and OLS

Fisher info/hessian

Delta method relies on more assumptions but is much faster than the bootstrap

But a closed form is easiest to work with, both for speed and intuition

# Appendix:

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

# Appendix: Multivariate Central Limit Theorem for multiple RVs $X_1, ..., X_k$

$MV Normal(\mu, Cov)$

$\mu = (\mathbb{E}[\mu_1], ..., \mathbb{E}[\mu_k]$)

$Cov = \frac{1}{n} \Sigma$, where $\Sigma$ is the covariance matrix of $X_1, ..., X_k$

---------------------------
---------------------------
---------------------------
---------------------------
---------------------------
---------------------------

# Draft

I need the standard error of something and it's not a "standard" statistic

so you need the variance of the sampling distribution of some statistic



Plan

✅ Understand the theory - what is the relevant theorem? where does it originate?

From wikipedia - https://en.wikipedia.org/wiki/Variance#Arbitrary_functions

If X is a random  variable, then:

$Var[f(X)] \approx (f'(\mathbb{E}[X]))^2 Var[X]$

(use this to derive a confidence interval and a hypothesis test by applying it to the sampling distribution)

multivariate version - https://en.wikipedia.org/wiki/Delta_method#Multivariate_delta_method

Double check but I think this _should_ work for X is multivariate, which becomes some scalar. like the ratio

$Var[f(X)] \approx\nabla f(X)^T Cov(X) \nabla f (X)$

$(1 \times k) (k \times k) (k \times 1)$

since it's a pretty elementary application of the taylor series, it was known sometime in the early 20th century

* Find a good real-life univariate example. maybe estimates of raw product and then production curve of finished product
* Do the math
* Sympy usage
* Numerical differentiation usage??
* What about the multivariate version? Maybe a simple example there too
* Write section 1: The common problem that the delta method solves, incl an example
* Write section 2: How to use it, in theory and with sympy/numerical analysis. Do simulation and confirm with bootstrap
* Write section 3: What about the multivariate case?
*  Other ways

Output
* When to use it
* How to use it
* A minimum of theory needed to understand it
* A clear worked example

Sources
* https://bookdown.org/ts_robinson1994/10EconometricTheorems/dm.html
* https://www.alexstephenson.me/post/2022-04-02-standard-errors-and-the-delta-method/
* https://en.wikipedia.org/wiki/Delta_method
* https://web.archive.org/web/20150525191234/http://data.math.au.dk/courses/advsimmethod/Fall05/notes/1209.pdf 
* https://egrcc.github.io/docs/math/all-of-statistics.pdf - 9.9, 9.10
