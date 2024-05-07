```python
import numpy as np

def block_bootstrap(ts, block_length):
    # Calculate the number of blocks based on the block length
    n_blocks = len(ts) // block_length
    
    # Check if the length of the time series is divisible by the block length
    assert len(ts) % block_length == 0, "Length of the time series is not divisible by the block length"
    
    # Generate the design matrix from the time series
    the_blocks = np.array([ts[i:(i + block_length)] for i in range(len(ts) - block_length + 1)])
    
    # Randomly select indices of blocks with replacement
    picked_blocks = np.random.choice(the_blocks.shape[0], size=n_blocks, replace=True)
    
    # Select the rows corresponding to the picked blocks
    x = the_blocks[picked_blocks, :]
    
    # Flatten the matrix into a vector
    x_vec = x.flatten()
    
    # Return the vector
    return x_vec

# Simulation study
from tqdm import tqdm
ts = np.cumsum(np.random.normal(0, 1, size=90))
block_length = 15

n_sim = 100000

block_sampled_paths = [block_bootstrap(ts, block_length) for _ in range(n_sim)]
iid_sampled_paths = [np.random.choice(ts, size=len(ts)) for _ in range(n_sim)]

from arch.bootstrap import StationaryBootstrap, optimal_block_length

print(optimal_block_length(ts))

arch_bs = StationaryBootstrap(block_length, ts)
arch_sampled_paths = [s[0][0] for s in arch_bs.bootstrap(n_sim)]

from matplotlib import pyplot as plt
import seaborn as sns

plt.plot(ts)
plt.plot(arch_sampled_paths[0])
plt.show()

from statsmodels.tsa.stattools import acf, pacf

plt.title('Sampled vs true partial autocorrelation functions')
[plt.plot(pacf(p, nlags=10), color='blue', alpha=.05) for p in arch_sampled_paths[:100]]
plt.plot(pacf(ts, nlags=10), color='orange')
plt.show()

plt.title('Simulated sampling distribution of mean')
sns.distplot([np.mean(p) for p in block_sampled_paths])
sns.distplot([np.mean(p) for p in iid_sampled_paths])

# TODO: What is the Type I error rate of each method? Does block have better than IID?
```

https://arch.readthedocs.io/en/latest/bootstrap/generated/arch.bootstrap.CircularBlockBootstrap.html#arch.bootstrap.CircularBlockBootstrap
