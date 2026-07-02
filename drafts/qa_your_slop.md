---
layout: post
title: "How to QA your slop and stop embarrassing yourself (using Bayesian statistics)"
author: "Louis Cialdella"
categories: posts
tags: [datascience]
image: how_to_train_your_dragon.jpg
---

# QA your slop, you're embarrassing yourself

# Basic failure rate statistics

# Bayesian analysis fast

Posterior = prior + data

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


# Posterior

Beta(a, b) and 95% CI on fail rate

<details>
  <summary> 🤔 What else can I do with the posterior? </summary>
  <table bgcolor="#235891" width="90%">
    <tr>
      <td>
        <p>sampling and differencing</p>
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