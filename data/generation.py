#!/usr/bin/env python
import random, math
import json

# Define a data structure
# two dimensional space, like a image, but only 2x2 pixel.

base_pattern = [
    [1,1,0,0],
    [1,0,1,0],
    [1,0,0,1],
]

def randomization(pattern: list) -> list:
    new_sample = pattern.copy()
    for i in range(len(new_sample)):
        value = new_sample[i]
        new_sample[i] = math.floor(abs(value - random.uniform( 0.0, 0.1 )) * 255)
    return new_sample

def convert_1d2d(one_d, rows, cols):
    if len(one_d) != rows * cols:
        raise ValueError("The length of the one-dimensional array does not match the specified dimensions.")
    return [one_d[i * cols:(i + 1) * cols] for i in range(rows)]

def debug_func():
    import matplotlib.pyplot as plt
    import numpy as np


    rand_data = randomization( 1, base_pattern[2] ) 
    np_array = np.array( convert_1d2d(rand_data, 2, 2), dtype=np.uint8 )

    plt.imshow(np_array, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')  # Turn off axis numbers
    plt.show()


def main():#DEBUG: bool) -> None:
    #if DEBUG:
    #    debug_func()
    #else:
    if True:
        data = []
        # generate 100 samples
        for i in range(0, 50):
            # runthought every pattern
            for pi, pattern in enumerate(base_pattern):
                # Randomize the pattern and store it with its label
                randomized_pattern = randomization(pattern)
                data.append({
                    'label': pi,
                    'pattern': randomized_pattern,
                })

        # Total: 100 * 3 samples
        f = open('data.json', 'w')
        random.shuffle(data)
        f.write(json.dumps( data, indent=4 ))
        f.close()

if __name__ == "__main__":
    import fire
    fire.Fire(main)
