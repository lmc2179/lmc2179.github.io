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

A VaR type problem?

For DES see this: https://www.youtube.com/watch?v=0KvA92ykPKI&ab_channel=UQComputingSociety
