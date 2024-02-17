# Questing for "the scikit learn of causal inference"

it has a simple modular workflow

it supports a wide variety of models under a common interface.

it solves a task people frequently have, ie E[y | X]. There are multiple examples of this in Casual Inference

# Obtaining the causal graph

There are many ways to do this

* Discovery algorithms
* Domain knowledge
* Experiments
* Econometric analysis
* Regression

Then represent it as networkx, maybe also summarize with a single number related to the arrow strength or elasticity or coefficient or interventional_samples

# An example workflow and library: DoWhy

DoWhy close reading - "An opinionated introduction to DoWhy"

https://www.youtube.com/watch?v=icpHrbDlGaw

I. Model a causal problem

Currently, dowhy doesn't really support discovery. Arrow strength is useful. Maybe include an example from something else here

II. Identify a target estimand under the model

III. Estimate causal effect based on the identified estimand

IV. Refute the obtained estimate

Pick an example including both a root cause analysis and an intervention analysis from the R data sets.

* House pricing? RCA: Why so expensive; Intervention: what if I made some renovation;

Assembling the causal graph

* The DAG (NetworkX)
* Root distributions and interior functional forms
* How do we deal with categorical variables? (Look at the docs for this)
* How to fit the model
* Root cause analysis howto
* Effect estimation
