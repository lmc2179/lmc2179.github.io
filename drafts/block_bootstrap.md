```python
import numpy as np

def block_bootstrap(ts, block_length):
    """
    https://www.stat.cmu.edu/~cshalizi/ADAfaEPoV/ADAfaEPoV.pdf s23.5.2, code example 34, with the help of chatgpt to translate from R
    """
    assert len(ts) % block_length == 0, "Length of the time series is not divisible by the block length"
    n_blocks = len(ts) / block_length  # Calculate the number of blocks based on the block length
    the_blocks = np.array([ts[i:(i + block_length)] for i in range(len(ts) - block_length + 1)])  # Generate blocks from the time series
    picked_blocks = np.random.choice(the_blocks.shape[0], size=n_blocks, replace=True)  # Randomly select indices of blocks with replacement
    x = the_blocks[picked_blocks, :]  # Select the rows corresponding to the picked blocks
    x_vec = x.flatten()  # Flatten the matrix into a vector
    return x_vec  # Return the vector

# Example usage
ts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
block_length = 6
result = rblockboot(ts, block_length)
print(result)
```
