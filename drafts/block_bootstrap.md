
Code below adapted from https://www.stat.cmu.edu/~cshalizi/ADAfaEPoV/ADAfaEPoV.pdf s23.5.2, code example 34, with the help of chatgpt to translate from R

```python
import numpy as np

def rblockboot(ts, block_length):
    """
    Generates a vector by randomly selecting blocks of a specified length from the given time series.

    Parameters:
        ts (list): The input time series.
        block_length (int): The length of each block to be selected.

    Returns:
        numpy.ndarray: A flattened vector containing the selected blocks.
    """
    # Calculate the number of blocks based on the block length
    n_blocks = len(ts) // block_length
    
    # Check if the length of the time series is divisible by the block length
    assert len(ts) % block_length == 0, "Length of the time series is not divisible by the block length"
    
    # Generate blocks from the time series
    the_blocks = np.array([ts[i:(i + block_length)] for i in range(len(ts) - block_length + 1)])
    
    # Randomly select indices of blocks with replacement
    picked_blocks = np.random.choice(the_blocks.shape[0], size=n_blocks, replace=True)
    
    # Select the rows corresponding to the picked blocks
    x = the_blocks[picked_blocks, :]
    
    # Flatten the matrix into a vector
    x_vec = x.flatten()
    
    # Return the vector
    return x_vec

# Example usage
ts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
block_length = 6
result = rblockboot(ts, block_length)
print(result)

```
