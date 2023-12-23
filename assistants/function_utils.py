import os
import math
import random

import ray
from typing import List, Dict

NUM_OF_CORES = None

def get_num_of_cores():
    """
    Returns the number of cores on the machine."""
    global NUM_OF_CORES
    if NUM_OF_CORES is None:
        NUM_OF_CORES = os.cpu_count()
    return NUM_OF_CORES 

def sampling_task(num_samples: int, task_id: int, verbose=True) -> int:
    num_inside = 0
    for i in range(num_samples):
        x, y = random.uniform(-1, 1), random.uniform(-1, 1)
        # check if the point is inside the circle
        if math.hypot(x, y) <= 1:
            num_inside += 1
    if verbose:
        print(f"Task id: {task_id} | Samples in the circle: {num_inside}")
    return num_inside

@ray.remote
def sample_task_distribute(sample_size, i) -> object:
    """
     A Ray remote function that samples points and returns the number of points """
    return sampling_task(sample_size, i)

def run_disributed(sample_size) -> List[int]:
    """
    Runs sampling_task_distribute in parallel using Ray."""
    # Launch Ray remote tasks in a comprehension list, each returns immediately with a future ObjectRef 
    # Use ray.get to fetch the computed value; this will block until the ObjectRef is resolved or its value is materialized.
    results = ray.get([
            sample_task_distribute.remote(sample_size, i+1) for i in range(get_num_of_cores())
        ])
    pi = calculate_pi(results, sample_size)
    return pi

def calculate_pi(results: List[int], sample_size:int) -> float:
    """
    Calculates pi from the results of the sampling tasks."""
    TOTAL_NUM_SAMPLES = get_num_of_cores() * sample_size
    total_num_inside = sum(results)
    pi = (total_num_inside * 4) / TOTAL_NUM_SAMPLES
    return pi

# Define a function which will be fed a dictionary with a key
# holding a list of randomly generated prime numbers between
# 2 and 100

def add_prime_numbers(p_numbers: Dict[str, List[int]]) -> int:
    return sum(p_numbers["prime_numbers"])

if __name__ == "__main__":
    # Run the sampling task locally
    sample_size = 10_000_000
    pi = run_disributed(sample_size)
    print(f"Estimated value of π is: {pi:5f}")
