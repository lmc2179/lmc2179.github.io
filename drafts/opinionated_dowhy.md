# Questing for "the scikit learn of causal inference"

it has a simple modular workflow

it supports a wide variety of models under a common interface.

it solves a task people frequently have, ie E[y | X]. There are multiple examples of this in Casual Inference

# An example workflow and library: DoWhy

DoWhy close reading - "An opinionated introduction to DoWhy"

https://www.youtube.com/watch?v=icpHrbDlGaw

I. Model a causal problem

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
