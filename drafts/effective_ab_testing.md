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

???

P-values/H0 issues: H0 isn't true, H0 isn't interesting, P-values run together power + effect - Gelman citation; even when they work, P-values only tell us about a non-zero effect, that's what "statistical significance" means

Beyond binary effect sizes: What is ES; unitless standardized measures of ES like Cohen's D as measures of effect vs "background noise"; what is "large" is still unclear (https://stats.stackexchange.com/questions/469203/what-is-the-intuition-behind-cohens-d, https://stats.stackexchange.com/questions/23775/interpreting-effect-size/87968#87968) and so I'd skip them


Effects large enough to care about: Org goals and the ROPE; work with your stakeholders!!



# An effective A/B test workflow: From design to decision

- Measure or guess the variance
- Elicit a ROPE
- Plan on a sample size which gets you a CI about the size of the ROPE
- Collect data and don't stop early
- Analysis: Compute delta CI, check if it is in ROPE; look at CI bounds, not the point estimate

# Appendix: Bayesian bonus:

P(delta)

Tripartition view

# Appendix: Halfnorm stuff

```python
import sympy as sm
from scipy.stats import halfnorm

x = sm.Symbol('x')
s = sm.Symbol('s')

f = (sm.sqrt(2/pi) * sm.exp(-x**2/2))/s

mean = sm.integrate(x*f, (x, 0, sm.oo))

var = sm.integrate(((x-mean)**2)*f, (x, 0, sm.oo))

print(mean.subs(s, 1).subs(pi, np.pi).evalf(), halfnorm(scale=1).mean())
print(var.subs(s, 1).subs(pi, np.pi).evalf(), halfnorm(scale=1).var())

s_in_terms_of_mu = sm.solve(mean-mu,s)[0]

s_when_mu_is_8 = s_in_terms_of_mu.subs(mu, 8).evalf()
```

--------------------------------------------------------


- A common anti-pattern is:
  - Collect $y^T, y^C$, calculate $\hat{\mu}^T, \hat{\mu}^C$
  - Run a T-test on $\mu^T = \mu^C$
  - If $p < .05$ and $\hat{\mu}^ > \hat{\mu}^C$, the test is a winner, and caused a $\frac{\hat{\mu}^T - \hat{\mu}^C}{\hat{\mu}^C}$ lift which your PM can put in the next update deck

This is an anti-pattern because it misleads us about the size of the treatment effect. The analysis does _not_ tell us about the size of the effect, aside from telling us that it is non-zero. The goal of A/B testing is to get a precise estimate of the treatment effect (Kruschke).

$H_0$ is never true or interesting

P-values run together precision and effect size

We need to define practical significance (ROPE) - usually we need to do ROPE elicitation
