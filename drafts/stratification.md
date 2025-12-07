# Stratification - CEM and HTE in Python

# The surprising effectiveness of stratification

Many research questions which aid in decision making are the question of "how does T affect Y, controlling for X". this is such a common and useful question to ask that it is occasionally mistaken to be the entire enterprise of causal inference

this problem has a familiar causal diagram, the confounding triangle (show it)

in my career, some examples I've seen over and over: how does using product T affect customer outcome Y (revenue, usually) when controlling for the mix of user types

more formally, we are usually looking to estimate the treatment effect; the average result of being moved from control to treatment

$\underbrace{\Delta}_\textrm{Treatment effect} = \underbrace{\mathbb{E}[y \mid T = 1]}_\textrm{Average outcome when treated} - \underbrace{\mathbb{E}[y \mid T = 0]}_\textrm{Average outcome when control} $

or the conditional treatment effect, which is the treatment effect given some specific information $X$ which we know, the conditional treatment effect:

$\underbrace{\delta(X)}_\textrm{Conditional Treatment effect} = \underbrace{\mathbb{E}[y \mid T = 1, X]}_\textrm{Average outcome when treated} - \underbrace{\mathbb{E}[y \mid T = 0, X]}_\textrm{Average outcome when control} $

stratification is basically always the first step for me when doing an analysis like this, even though there are lots of directions it might take after that

# Coarsening and stratifying

# Estimating the ATE with matching

# Building an HTE model