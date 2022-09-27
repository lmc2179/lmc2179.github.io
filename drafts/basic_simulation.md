This is incremental time progression, not DES https://en.wikipedia.org/wiki/Discrete-event_simulation

```python
def gen_initial():
  return {} # TODO: Set initial conditions
  
def get_sim_length():
  pass # Return a positive integer
  
def sim_next(history, t):
  pass # Look at all of history and simulate the next time step, which is t

def run_simulation():
  S = [gen_initial()]
  T = get_sim_length()
  for t in range(1, T+1):
    S.append(sim_next(S, t]))
  return pd.DataFrame(S) # Index is time
```
For example above: Walk w/ changing drift est. from data

For DES see this: https://www.youtube.com/watch?v=0KvA92ykPKI&ab_channel=UQComputingSociety

Possible example 1:

Idea: Get
[house data](https://en.wikipedia.org/wiki/List_of_United_States_House_of_Representatives_elections,_1856%E2%80%93present)
[pres data](https://en.wikipedia.org/wiki/List_of_United_States_presidential_elections_by_popular_vote_margin)
with [wiki package](https://pypi.org/project/Wikipedia-API/)

Hypothesis: Midterms are bad for the current president; incumbents tend to win; non-incumbent races are a toss-up
Use ABC! Find rules that agree with the actual path 80% of the time

Possible example 2: A VaR simulation problem, maybe on one of these data sets: https://github.com/lmc2179/lmc2179.github.io/blob/master/drafts/random_walk.md

Possible example 3: A problem requiring strong assumptions, like a model of how a firm grows; lets us see how sensitive we are the the assumptions

Possible example 4: A counterfactual example; how does the time series change if I intervene at time X
