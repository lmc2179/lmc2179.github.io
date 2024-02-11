Basic pid control and a simulated example - a practical guide in python

Idea: we often need to keep a system at some value. ensuring we have enough money in the budget, enough machines to handle requests, etc

example: buying coffee beans and keeping a buffer

We have a possible error (needed beans - actual beans) which we want to keep at zero. we have a lever (buying beans) which we should use when the error is large.

error sign is positive = more beans needed than actual --> lever sign should be positive (buy more beans)

The error is the BEAN GAP

Three components:

1. P: We should correct more strongly when the most recent error was large. We should buy a lot of beans when the BEAN GAP is large. 
2. I: We should correct more strongly when the error has been large for a long time. We should buy a lot of beans when the BEAN GAP has been large for a long time.
3. D: We should correct more strongly when the error is increasing quickly. We should buy a lot of beans when the the size of the BEAN GAP increased yesterday compared to the day before.

$Correction(t+1) = f_P(e(t)) + f_I(\sum_{i=1}^t e(i)) + f_D(e(t) - e(t-1))$

Simulated example where demand is sampled randomly, maybe a random walk with 7-day correlation?

this a simple kind of reinforcement learning - we are learning a very simple policy which responds to conditions and minimized error.
