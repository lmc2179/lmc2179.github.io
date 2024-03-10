Of all the possible causes of X, which ones are most important?

estimate the arrow strength! see dowhy link

1. Collect all the potential causes (treatments) and their confounders
2. Make a graph - probably it's a version of the confounding triangle, where each node is either a cause, confounder, or outcome
3. train models of output = \hat{f}(causes, confounders). compare with baseline, take the best ones
4. for each one, estimate arrow strength by simulating removing the arrow from cause_i -> outcome. Do this by shuffling

Shuffling method

* $\epsilon_{baseline}$ - highest error, from the dummy model
* $\epsilon_{complete}$ - model with all the features, usually the lowest error
* $\epsilon_{minus \ i}$ - model without feature i, somewhere in the middle

then compute the importance $\Delta$ , which is _how much error comes from removing the feature vs the largest possible error_

$\Delta = \frac{$\epsilon_{minus \ i} - $\epsilon_{complete}}{$\epsilon_{baseline} - \epsilon_{complete}}$

the max importance is $\Delta = 1.0$, when removing the feature increases the error as much as removing all the features

train the model, then compute the importance for each feature using the model. interpretation is that this is the arrow strength _according to this particular model_
