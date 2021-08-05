Effective A/B testing is about practical significance, not statistical significance

# Why we do A/B testing, and why we often do it badly

We run an A/B test to learn how a change in the user experience affects user behavior. Since a well-planned A/B test can provide a definitive estimate of the causal effect of the change, it gives decision makers a set of facts to base decisons in. It creates organizational consensus.

Companies run A/B tests to see **how changes in the user experience affect user behavior**. They split users between the old, "control" experience and a new "treatment" experience, providing an estimate of how the change affects one or more measured **outcome metrics**. For example, ecommerce retailers like Amazon provide recommendations to users in the hope of increasing revenue per user. They might A/B test a new recommendation alongside an old one, to estimate the impact of the new algorithm on revenue. **Because they randomize away sources of variation other than the intervention, A/B tests provide a high-quality estimate of the causal effect of the change.** This clarifies the decision of whether to switch everyone over to the new algorithm. This post is about how to turn the test data into a decision: should we change algorithms or not?

A common **anti-pattern** I've observed is a workflow that goes like this:
  - Collect your data from the treated and control groups, and calculate the treatment and control averages per user (such as revenue per user). Since we're fancy data scientists who need to justify our salaries, we'll call the treatment and control averages estimated from the data $\hat{\mu}^T$ and  $\hat{\mu}^C$
  - Check if $\hat{\mu}^T > \hat{\mu}^C$. If not, call the test a failure.
  - Run a T-test on $H_0:\mu^T = \mu^C$. The idea here is that we'd like to be able to reject the null hypothesis that they're the same, ie that the treatment had no effect.
  - If $p < .05$ (or your favorite $\alpha$) and $\hat{\mu}^ > \hat{\mu}^C$, the test is a winner! You say it caused a $\hat{\mu}^T - \hat{\mu}^C$ increase in revenue, which your PM can put in the next update deck.

I call this anti-pattern **P-value sanctification**. The internal logic of this approach goes something like this: if $p < .05$, then the observed treatment effect of $\hat{\mu}^T - \hat{\mu}^C$ is "real", in the sense that it's not noise. Unfortunately, this interpretation isn't quite right.

# Problems with P-value sanctification, and some solutions

Before I point out the issues with this workflow, I'll note some _good_ things about the above method, which might explain why it's commonly observed in the wild.

1. It is easy to automate: each step involves a well-defined calculation that doesn't need a human in the loop. 
2. It provides an unambiguous decision process. Once we collect the data, we can simply compute $\hat{\mu}^T$, $\hat{\mu}^C$, and $p$, and we'll know whether we should switch to the new algorithm or not.

These are good things! As we try to improve this approach, we'll preserve these properties.

Let's turn now to the **flaws of this approach**:

1. The P-value analysis does not actually tell us about the magnitude of the lift. We only tested the hypothesis that the difference between treatment and control isn't _exactly_ zero. If we want to understand the size of the treatment effect, we should put error bars around it.
2. An increase which is statistically significant may not be **practically significant**. Even if we reject $H_0$, meaning that we think treatment provides more revenue than control, it can still be true that the increase is to small to make any practical difference to the success of the business. That is, the increse can be non-zero, but still be too small to matter to our stakeholders.

The first issue can be solved simply by reporting the estimated treatment effect with its error bars. The second problem, though, is trickier - it requires us to answer a substantive question about the business. Specifically, we need to answer the question: **what size of treatment effect _would_ be practically significant?** 

???

P-values/H0 issues: H0 isn't true, H0 isn't interesting, P-values run together power + effect - Gelman citation; even when they work, P-values only tell us about a non-zero effect, that's what "statistical significance" means

## How effect size measures like Cohen's $d$ try to solve this problem, and their pitfalls

Beyond binary effect sizes: What is ES; unitless standardized measures of ES like Cohen's D as measures of effect vs "background noise"; what is "large" is still unclear (https://stats.stackexchange.com/questions/469203/what-is-the-intuition-behind-cohens-d, https://stats.stackexchange.com/questions/23775/interpreting-effect-size/87968#87968) and so this is not an ideal solution

$d = \frac{\mu^T - \mu^C}{\sigma}$; numerator is effect, denominator is "background noise" the normal amount of variation

this is a step in the right direction, in that it compares the effect with a benchmark and doesn't depend on sample size

similar to R-squared interpretability; R-squared is sort of like an effect size measure since we're comparing vs the regression background noise

## ROPE

Effects large enough to care about: Org goals and the ROPE; work with your stakeholders!!

How to elicit a ROPE - bracket it with "Would it be worth it to switch if the increase were x"; start too small and too large, find the r where it switches

# An effective A/B test workflow: From design to decision

- Measure or guess the variance
- Elicit a ROPE
- Plan on a sample size which gets you a CI about the size of the ROPE
- Collect data and don't stop early
- Analysis: Compute delta CI, check if it is in ROPE; look at CI bounds, not the point estimate

0. Make some assumptions: balanced samples sizes, equal variance in control and treatment, CLT, positive change is better
1. Measure or guess the variance of the goal metric for the test. If there's no historical data, consider a pessimistic case based on what you know (ie, Bernoulli example)
2. Work with stakeholders to select a ROPE
3. Compute the sample size
4. Collect the data
5. Calculate the treatment effect $\hat{\delta} = \hat{\mu^T} - \hat{\mu^C}$, $\hat{SE} = \sqrt{\hat{SE}_T^{2} + \hat{SE}_C^{2}}$

# Appendix: Computing sample sizes when a ROPE is used and sample sizes are balanced

- Let $[0, r]$ be the ROPE
- Let $\sigma_T^2, \sigma_C^2$ be the variances in each bucket
- Let $z_\alpha$ be the critical value, like 1.96
- Let $n_T, n_C$ be the sample sizes in each bucket
- Then $SE(\hat{\delta}) = \sqrt{\frac{\sigma_T^2}{n_T} + \sigma_C^2}{n_C}}$
- We want: $2z_\alpha SE(\hat{\delta}) = r$
- Let variances and samples sizes be equal
- Solve for $n$

# Appendix: Bayesian bonus:

P(delta)

Tripartition view
