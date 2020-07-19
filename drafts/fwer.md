# Bonferroni, Bonferroni-Holm - Methods that control the FWER and their costs

TL;DR - Multiple hypothesis testing shows up frequently in practice, but needs to be handled carefully. Common methods like the Bonferroni correction are useful quick fixes, but they have important drawbacks that users should understand. *What do methods like Bonferroni and Bonferroni-Holm do, and why do they work? What alternatives are there?*

## Multiple testing is everywhere, and can be tricky

Any time we compute a P-value to test a hypothesis, we run the risk of being unlucky and rejecting the null hypothesis when it is true. This is called a "false positive" or "Type I error", and we carefully construct our hypothesis tests so that it only happens 1% or 5% of the time - more generally, that it happens $\alpha%$ of the time. There's an analogous risk for confidence intervals; we might compute an interval that doesn't cover the true value, though again this shouldn't happen too often.

Almost as soon as we start thinking of data analysis as opportunities to test hypotheses, we encounter situations where we want to test multiple hypotheses at once. For example:

- When we have an A/B test with more variants than just "Test" and "Control" and we want to compare them all to each other
- When we want to compare the out-of-sample error rates of multiple machine learning models to each other
- When we want to look at the significance of many coefficients in a multiple regression at the same time

However, this needs to be handled more carefully than simply computing a pile of P-values and rejecting whichever ones are smaller than $\alpha$. For any given hypothesis test, we have a false positive rate of $\alpha$; we spend a lot of time making sure our test procedures keep the likelihood of a false positive below $\alpha$. But this guarantee *doesn't* hold up when we test multiple hypotheses.

For example, let's say we examine two distinct null hypotheses, which are both true. Even though $H_0$ is true, we run an $\alpha$ probability risk of rejecting each one. If the test statistics are uncorrelated, this means that our chance of *any* false positives is $1 - (1-\alpha)^2$. If $\alpha = .05$, for example, this means that we have a 9.75% chance of at least one false positive. This problem gets worse the more simultaneous tests we run - if we run enough tests, we'll eventually find a null hypothesis to reject, even when all the nulls are true. This has the unfortunate implication that the more questions we ask, the more likely we are to get an answer that rejects a null hypothesis, even when all the null hypotheses are true. To borrow a quote without context, "seek, and ye shall find" - though what you find may be spurious.

Here's a stylized example of the problem:

![A classic example](https://imgs.xkcd.com/comics/significant.png)

This is a classic example of multiple testing error - if you test 20 times at the 5% significance level, you've got a much better chance of finding a spurious non-null relationship, and all your usual guarantees about "only happening 5% of the time" are no longer valid.

## A simulation example: Estimating multiple means

Let's take a look at a simulated example to see this kind of problem in action. Then, once we have some possible solutions, it will be easy to see

We're going to simulate $n = 30$ draws of 5 random variables, which we'll call $X_1, X_2, X_3, X_4, X_5$. All of these variables will be normally distributed, and have a variance of 1. The first 4 will have a mean of zero, and last one will have a mean of one half. We'll test the 5 hypotheses that the mean of each individual group is zero. This means that in this example, four of the five null hypotheses are true (the null is true for all the variables but X5).

First, let's write a function to simulate the data:

```python
from scipy.stats import ttest_1samp, sem, norm

def generate_dataset():
  n = 30 # As we know from Intro Stats, 30 is infinity
  X1, X2, X3, X4, X5 = norm(0, 1), norm(0, 1), norm(0, 1), norm(0, 1), norm(.5, 1)
  X1_samples, X2_samples, X3_samples, X4_samples, X5_samples = X1.rvs(n), X2.rvs(n), X3.rvs(n), X4.rvs(n), X5.rvs(n)
  return X1_samples, X2_samples, X3_samples, X4_samples, X5_samples
```

And create a function to calculate which of the P-values are significant:

```python
def calculate_significance(X1_samples, X2_samples, X3_samples, X4_samples, X5_samples, alpha):
  p_values = np.array([ttest_1samp(X1_samples, 0)[1], 
                                   ttest_1samp(X2_samples, 0)[1],
                                   ttest_1samp(X3_samples, 0)[1],
                                   ttest_1samp(X4_samples, 0)[1],
                                   ttest_1samp(X5_samples, 0)[1]])
  is_significant = p_values < alpha
  return is_significant
```

Which we can run in order to create a whole lot of simulated datasets and whether we reject our five null hypotheses for each one:

```python
n_sim = 10000
significant_simulations = list()
for _ in range(n_sim):
  significant_simulations.append(calculate_significance(*generate_dataset(), alpha=.05))

significant_simulations = np.array(significant_simulations)
```

Great. That's 10000 simulations of the kinds of inferences we'd make in this situation.

```python
X1_rejected = significant_simulations[:,0]
X2_rejected = significant_simulations[:,1]
X3_rejected = significant_simulations[:,2]
X4_rejected = significant_simulations[:,3]
X5_rejected = significant_simulations[:,4]

print('Proportion of times that we rejected H0 for X1: {0}'.format(np.mean(X1_rejected)))
print('Proportion of times that we rejected H0 for X2: {0}'.format(np.mean(X2_rejected)))
print('Proportion of times that we rejected H0 for X3: {0}'.format(np.mean(X3_rejected)))
print('Proportion of times that we rejected H0 for X4: {0}'.format(np.mean(X4_rejected)))
print('Proportion of times that we rejected H0 for X5: {0}'.format(np.mean(X5_rejected)))

print('Proportion of times we falsely rejected any null hypotheses: {0}'.format(np.mean(X1_rejected | X2_rejected | X3_rejected | X4_rejected)))
```

Output:
```
Proportion of times that we rejected H0 for X1: 0.0516
Proportion of times that we rejected H0 for X2: 0.0515
Proportion of times that we rejected H0 for X3: 0.05
Proportion of times that we rejected H0 for X4: 0.051
Proportion of times that we rejected H0 for X5: 0.7473
Proportion of times we falsely rejected any null hypotheses: 0.1905
```

Lets break that down:
For each individual true null hypothesis, we rejected the null hypothesis around 5% of the time, the expected false positive rate.
For X5, which does *not* have a mean of zero, we correct reject the null about 75% of the time - so we have about 75% power for X5 with this mean.
While each individual test retains its correct false positive rate, we have a high probability of reporting at least one false positive. This seems like a problem - our analysis will report at least one false positive almost 20% of the time!

This last quantity - the proportion of times we falsely reject *any* null hypotheses - has a special name. We call it the **Family-wise Error Rate**, or the **FWER**. It's one way to define the "false positive rate" for multiple experiments. Usually, when we refer to "controlling the FWER", we mean a testing procedure which would keep the FWER below $\alpha$. 

FWER isn't the only way to think about false positives in multiple tests. Another option is to try and control the number of rejected hypotheses which are false positives (defined as Number of False rejections / Number of all rejections). This quantity is called the "False Discovery Rate" (FDR), and it's worth thinking about as a less strict alternative to the FWER. But that's a story for another time. For now, let's talk about how we can control the FWER using a method familiar to many who have encountered this problem before - the Bonferroni correction.

## One solution : Set the bar higher, control the Family-Wise Error rate

The fundamental issue is that when we test more things, we're more likely to get at least one false positive. One way to solve this is to set the bar higher - if we are less likely to reject the null hypothesis overall, we'll be less likely to make a false positive.

How much higher should we set the bar? That is, how should we change $\alpha$ in order to account for the increased false positive rate?

## A common fix: Bonferroni correction to control the FWER

```python
def calculate_significance_bonferroni(X1_samples, X2_samples, X3_samples, X4_samples, X5_samples, alpha):
  p_values = np.array([ttest_1samp(X1_samples, 0)[1], 
                                   ttest_1samp(X2_samples, 0)[1],
                                   ttest_1samp(X3_samples, 0)[1],
                                   ttest_1samp(X4_samples, 0)[1],
                                   ttest_1samp(X5_samples, 0)[1]])
  is_significant = p_values < alpha/5.
  return is_significant

n_sim = 10000
significant_simulations = list()
for _ in range(n_sim):
  significant_simulations.append(calculate_significance_bonferroni(*generate_dataset(), alpha=.05))

significant_simulations = np.array(significant_simulations)

X1_rejected = significant_simulations[:,0]
X2_rejected = significant_simulations[:,1]
X3_rejected = significant_simulations[:,2]
X4_rejected = significant_simulations[:,3]
X5_rejected = significant_simulations[:,4]

print('Proportion of times that we rejected H0 for X1: {0}'.format(np.mean(X1_rejected)))
print('Proportion of times that we rejected H0 for X2: {0}'.format(np.mean(X2_rejected)))
print('Proportion of times that we rejected H0 for X3: {0}'.format(np.mean(X3_rejected)))
print('Proportion of times that we rejected H0 for X4: {0}'.format(np.mean(X4_rejected)))
print('Proportion of times that we rejected H0 for X5: {0}'.format(np.mean(X5_rejected)))

print('Proportion of times we falsely rejected any null hypotheses: {0}'.format(np.mean(X1_rejected | X2_rejected | X3_rejected | X4_rejected)))
```

Output:
```
Proportion of times that we rejected H0 for X1: 0.0107
Proportion of times that we rejected H0 for X2: 0.0108
Proportion of times that we rejected H0 for X3: 0.0087
Proportion of times that we rejected H0 for X4: 0.0103
Proportion of times that we rejected H0 for X5: 0.5082
Proportion of times we falsely rejected any null hypotheses: 0.0398
```

The FP rate for each individual test is now $\frac{\alpha}{5} = \frac{.05}{5} = .01$.
The FWER was successfully reduced to below $\alpha$.
The power was reduced - we only successfully rejected the null for X5 50% of the time, compared with 75% of the time without the correction.

## The tradeoff of Bonferroni: FWER control at the cost of reduced power

## Why does the Bonferroni correction work?

We'd like to show that when we set the significant level to $\frac{\alpha}{m}$, the FWER is not more than $\alpha$. We can break down an FWER violation as the union of all events where one P-value of a true $H_0$ is less than the significance level. In the following, let's define:

- $m$, the total number of hypotheses we want to test
- $m_0$, the total number of null hypotheses which are true
- $p_i$, the p-value for hypothesis $i$
- $\alpha$, the significance level set by the analyst

|---|---|
|  $\mathbb{P} (\bigcup_{i=1}^{m_0} p_i \leq \frac{\alpha}{m})$   | This is the definition of the FWER under the Bonferroni-correction. |
| $\leq \sum_{i=1}^{m_0} \mathbb{P}(p_i \leq \frac{\alpha}{m})$     |  Union bound (no assumptions, add endnote) |
|  $= m_0 \frac{\alpha}{m}$  |  $\mathbb{P}(p \leq X)$ when $H_0$ is true |
| $\leq m \frac{\alpha}{m} = \alpha $ | Because $m_0 \leq m$ |

CI variant endnote

$$\mathbb{P} (\bigcup_{i=1}^{m} \mu_i \notin CI_{\frac{\alpha}{m}}(X_i) )$$

$$\leq \sum_{i=1}^{m} \mathbb{P}(\bigcup_{i=1}^{m} \mu_i \notin CI_{\frac{\alpha}{m}}(X_i) )$$

$$= m \frac{\alpha}{m} = \alpha$$

## What about confidence intervals?

So far we've talked about simultaneously testing a number of hypotheses by computing a number of P-values. You might wonder whether the procedure is any more complicated if we're interested in simultaneous confidence intervals, rather than P-values. It turns out that the Bonferroni procedure works without any real change if you're computing confidence intervals - all you need to do is change the significance level of all your intervals to $\frac{\alpha}{m}$.

## A more powerful procedure for P-values: Bonferroni-Holm

```python
is_significant = multipletests(p_values, method='holm', alpha=.05)[0]
```

Simulation w/ BH produces only marginally more power in this case, but it can be meaningful

Doing it in one line of Statsmodels

No reason not to do it when you're doing P-value stuff

## Why does Bonferroni-Holm work?

## FWER control procedures other than Bonferroni and Bonferroni-Holm

Tukey
Dunnet
MCB

## What alternatives might we use instead of the FWER? The FDR and Hierarchical model approaches
