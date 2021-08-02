Effective A/B testing is about practical significance, not statistical significance

# Why we do A/B testing, and why we often do it badly

We run an A/B test to learn how a change in the user experience affects user behavior. Since a well-planned A/B test can provide a definitive estimate of the causal effect of the change, it gives decision makers a set of facts to base decisons in. It creates organizational consensus.

Example: New recommendation algorithm for an ecomm site - does it do anything? does it do enough to care about?

Does the new feature improve revenue per user enough for us to care about?

A common workflow of the "P-value sanctification" anti-pattern is:
  - Collect $y^T, y^C$, calculate $\hat{\mu}^T, \hat{\mu}^C$
  - Run a T-test on $\mu^T = \mu^C$
  - If $p < .05$ and $\hat{\mu}^ > \hat{\mu}^C$, the test is a winner, and caused a $\frac{\hat{\mu}^T - \hat{\mu}^C}{\hat{\mu}^C}$ lift which your PM can put in the next update deck

This is an anti-pattern because it misleads us about the size of the treatment effect. The analysis does _not_ tell us about the size of the effect, aside from telling us that it is non-zero. The goal of A/B testing is to get a precise estimate of the treatment effect (Kruschke).

# Problems with P-values, and some solutions

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
