# Archetypes of analysis: Description, Statistical Inference, Causal Inference

Its tempting do be ambitious and set out from the get go to have your analysis be a causal, scientific analysis, but it pays to start simple and work your way up to that

It's kind of an agile method: Collect data ↔️ Description ↔️ Statistical Inference ↔️ Causal Inference. Often you find gaps that prevent you from taking the next step up the ladder. the solution is to go get more data in those cases (link shoe leather paper)

What does each step _require_? Can your knowledge of the data and the DGP really provide it?

https://www.public.asu.edu/~gasweete/crj604/readings/2010-Berk%20(what%20you%20can%20and%20can%27t%20do%20with%20regression).pdf

> By default, the true enterprise is description. Most everything else is puffery.

The original article is about regression analysis; lets talk about both classical and black box regression models

# Level I: Describing the data

Observed distributional facts

What kinds of models of E[y | X] fit

# Level II: Statistical inference about a population

Statistical significance of differences (between subgroups, treated/untreated, etc)

Whether we're pretty sure that for a model that fits well, some coefficient should be zero. 

Bootstrapping. Bootstrapping makes you think about whether you're really simulating the sampling process

# Level III: Causal inference about an intervention

Causal Inference requires causal assumptions

DAGs and unconfoundedness
