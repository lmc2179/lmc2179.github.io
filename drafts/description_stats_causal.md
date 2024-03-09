# Archetypes of analysis: Description, Statistical Inference, Causal Inference

Its tempting do be ambitious and set out from the get go to have your analysis be a causal, scientific analysis, but it pays to start simple and work your way up to that

It's kind of an agile method: Collect data ↔️ Description ↔️ Statistical Inference ↔️ Causal Inference. Often you find gaps that prevent you from taking the next step up the ladder. the solution is to go get more data in those cases (link shoe leather paper)

What does each step _require_? Can your knowledge of the data and the DGP really provide it?

https://www.public.asu.edu/~gasweete/crj604/readings/2010-Berk%20(what%20you%20can%20and%20can%27t%20do%20with%20regression).pdf

> By default, the true enterprise is description. Most everything else is puffery.

The original article is about regression analysis; lets talk about both classical and black box regression models

https://psychology.okstate.edu/faculty/jgrice/psyc5314/Freedman_1991A.pdf

> As a piece of statistical technology, Table 1 is by no means remarkable. But the story it tells is very persuasive. The force of the argument results from the clarity of the prior reasoning, the bringing together of many different lines of evidence, and the amount of shoe leather Snow was willing to use to get the data.
> Snow did some brilliant detective work on nonexperimental data. What is impressive is not the statistical technique but the handling of the scientific issues. He made steady progress from shrewd observation through case studies to analysis of ecological data. In the end, he found and analyzed a natural experiment.

# Level I: Describing the data

Description answers questions like: How many poll respondents voted for X

Observed distributional facts

What kinds of models of E[y | X] fit this data well

# Level II: Statistical inference about a population

Point estimates, confidence intervals, postriors, sampling distributions

Berk just talks about SI, but Prediction is also inference about a population, ie inferring the the conditional mean E[y | X] - a lot of ML activity actually lives here

SI/P answers questions like: What is the political opinion of all the people in the state? Which people will show up on election day?

Statistical significance of differences (between subgroups, treated/untreated, etc)

Whether we're pretty sure that for a model that fits well, some coefficient should be zero. 

Bootstrapping. Bootstrapping makes you think about whether you're really simulating the sampling process

# Level III: Causal inference about an intervention

CI answers questions like: What things cause people to vote for X, such that we could intervene and affect their willingness to vote for X

Causal Inference requires causal assumptions

DAGs and unconfoundedness
