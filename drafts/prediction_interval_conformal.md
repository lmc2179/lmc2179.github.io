# Why are PIs useful? Real-life applications

It's useful to make predictions about the range of plausible outcomes

# Some ways of getting PIs

https://lmc2179.github.io/posts/confidence_prediction.html

https://lmc2179.github.io/posts/quantreg_pi.html

# Black box PIs with conformal inference

user choices (after MAPIE paper s2.1)

1. pick conformity score
2. How to generate the out-of-sample predictions (jackknife, CV, etc)
3. Risk level \alpha

# The key idea in conformal inference: PIs in arbitrary spaces based where conformity=distance

Distance framing of conformal inference

1. Pick a distance measure in the Y-space
2. Generate OOS predictions
3. Look at how close predictions "usually" are to the actual values
4. Make a prediction; points which are the "usual" distance from the prediction are included in the PI

# MAPIE example for regression: Training and prediction

# how it works for stuff other than regression

classification intuition

time series/enbPI

# Evaluating PI models

hit rate/coverage

https://stats.stackexchange.com/questions/465799/testing-for-clairvoyance-or-performance-of-a-model-where-the-predictions-are-i/465808#465808

which is introduced in section 6.2 of https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf


# Appendix - relevant papers

gentle introduction - https://arxiv.org/pdf/2107.07511.pdf

mapie - https://arxiv.org/abs/2207.12274
