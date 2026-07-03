---
layout: post
title: "How to QA your slop and stop embarrassing yourself (using Bayesian statistics)"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: how_to_train_your_dragon.jpg
---

# QA your slop, you're embarrassing yourself

As far as I can tell, every tech company around has gone absolutely slop wild.

If your brain is still working, you probably realize that you need to have some actual people check the output from time to time. But human attention is expensive. You can take batches of output and sample from them. This is about how to use statistics to figure out if your failure rate is good enough

* What is the failure rate for this batch of outputs?
* Is the failure rate within acceptable bounds?
* How many absolute failures are likely given the test cases you saw?

# Bayesian analysis fast

Posterior = prior + data

Bayesian analysis is a little three-step dance that goes like this:
* Pick a prior
* Observe the data
* Update the prior to get the posterior. Use the posterior to calculate the probability  $\mu < 5\%$.

<details>
  <summary> 🤔 Wait, what's a probability distribution? </summary>
  <table bgcolor="#235891" width="90%">
    <tr>
      <td>
        <p>probability distributions</p>
            <details>
            <summary> 🤔 more references </summary>
            <table bgcolor="#8f4040" width="90%">
                <tr>
                <td>
                    <p>references</p>
                </td>
                </tr>
            </table>
            </details>
      </td>
    </tr>
  </table>
</details>

# Prior

Beta(1, 1) prior

$\underbrace{\mu}_{\text{Our prior on the rate}} \sim \underbrace{Beta(\alpha_0, \beta_0)}_{\text{is a Beta distribution with params } \alpha_0, \beta_0}$

You can interpret the prior parameters as "hypothetical data" that summarizes your beliefs about the rate. 

<details>
  <summary> 🤔 Why Beta(1, 1)? What about other choices </summary>
  <table bgcolor="#235891" width="90%">
    <tr>
      <td>
        <p>other choices</p>
            <details>
            <summary> 🤔  Tell me more </summary>
            <table bgcolor="#8f4040" width="90%">
                <tr>
                <td>
                    <p>kerman, elicitation</p>
                </td>
                </tr>
            </table>
            </details>
      </td>
    </tr>
  </table>
</details>

# Update

Update equation

<details>
  <summary> 🤔 Why is the posterior still a beta distribution? </summary>
  <table bgcolor="#235891" width="90%">
    <tr>
      <td>
        <p>conjugate prior</p>
            <details>
            <summary> 🤔  What if I can't use a conjugate prior </summary>
            <table bgcolor="#8f4040" width="90%">
                <tr>
                <td>
                    <p>pymc</p>
                </td>
                </tr>
            </table>
            </details>
      </td>
    </tr>
  </table>
</details>

$\underbrace{\mu \mid y, n}_{\text{The posterior of the rate given the data}} \sim \underbrace{Beta(\alpha_0 + y, \beta_0 + N - y)}_{\text{is given by this beta distribution}}$



# Posterior analysis

Beta(a, b) and 95% CI on fail rate






<details>
  <summary> 🤔 What else can I do with the posterior? </summary>
  <table bgcolor="#235891" width="90%">
    <tr>
      <td>
        <p>sampling and differencing</p>

from scipy.stats import beta

y = 40
n = 1000

a_0 = 1./3
b_0 = 1./3

posterior = beta(a_0 + y, b_0 + n - y)

n_simulations = 100000
posterior_samples = posterior.rvs(n_simulations)

print('Monte carlo estimate of P(Rate) < 5%: ', sum(posterior_samples < .05) / len(posterior_samples))

print('CDF(5%) = ', posterior.cdf(.05))
            <details>
            <summary> 🤔  What about </summary>
            <table bgcolor="#8f4040" width="90%">
                <tr>
                <td>
                    <p>???</p>
                </td>
                </tr>
            </table>
            </details>
      </td>
    </tr>
  </table>
</details>

# Posterior predictive

Beta binomial (n, a, b) and failure count

<details>
  <summary> 🤔 How could I use this to estimate the cost of failure </summary>
  <table bgcolor="#235891" width="90%">
    <tr>
      <td>
        <p>sampling simulation</p>
            <details>
            <summary> 🤔  What about </summary>
            <table bgcolor="#8f4040" width="90%">
                <tr>
                <td>
                    <p>???</p>
                </td>
                </tr>
            </table>
            </details>
      </td>
    </tr>
  </table>
</details>