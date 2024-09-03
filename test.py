import numpy as np

# Original array
array = np.array([
    [ 1, 2, 3, 4, 5],
    [ 3, 4, 5, 6, 7],
    [ 6, 7, 8, 9, 10],
    [ 8, 9, 10, 11, 12],
    [11, 12, 13, 14, 15],
    [13, 14, 15, 16, 17],
    [16, 18, 18, 19, 20]
])

# Define window size and overlap
window_size = 10
step_size = 3  # This corresponds to the overlap of 1 between consecutive windows

# Create the new array by concatenating overlapping windows
new_array = np.array([np.concatenate(array[i:i+window_size].flatten()) for i in range(0, array.shape[0] - window_size + 1, step_size)])

print(new_array)







