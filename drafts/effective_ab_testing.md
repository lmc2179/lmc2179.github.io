A/B testing is about practical significance, not statistical significance

- A common anti-pattern is:
  - Collect $y^T, y^C$, calculate $\hat{\mu}^T, \hat{\mu}^C$
  - Run a T-test on $\mu^T = \mu^C$
  - If $p < .05$ and $\hat{\mu}^ > \hat{\mu}^C$, the test is a winner, and caused a $\frac{\hat{\mu}^T - \hat{\mu}^C}{\hat{\mu}^C}$ lift which your PM can put in the next update deck

This is an anti-pattern because it misleads us about the size of the treatment effect. The analysis does _not_ tell us about the size of the effect, aside from telling us that it is non-zero. The goal of A/B testing is to get a precise estimate of the treatment effect (Kruschke).

$H_0$ is never true or interesting

P-values run together precision and effect size

We need to define practical significance (ROPE) - usually we need to do ROPE elicitation
