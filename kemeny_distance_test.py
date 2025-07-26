"""
Test and benchmarking script for Kemeny distance algorithms.

This script verifies the correctness of the efficient O(N log N) Kemeny
distance algorithm against a naive O(N^2) implementation. It also benchmarks
the run time of both algorithms and generates a plot to visualise their
time complexity.

To run this script, make sure 'kemeny_distance.py' is in the same directory.
"""

import math
import random
import time
import matplotlib.pyplot as plt
from kemeny_distance import (
    compute_kemeny_distance, 
    compute_normalised_kemeny_distance,
    naive_kemeny_distance,
    naive_normalised_kemeny_distance
)

# Set random seed for reproducibility
RANDOM_SEED = 555

# Define the prediction semantics vector size for performance testing
# Starts at 10, goes up to 1000 with a step of 20
PERFORMANCE_TEST_SIZES = range(10, 1000, 20)

# Number of iterations for each performance test to get a stable average
TIME_ITERATIONS = 1

# Define prediction semantics vector size for correctness check
N = 20
# Define the number of test cases for correctness check
N_TEST_CASES = 100

def generate_random_prediction_semantics(n): 
    """Generate a list of 'n' random float with 1 decimal place to promote ties"""
    return [round(random.random(), 1) for _ in range(n)]

def run_correctness_check():
    """Verify that the efficient and naive algorithms produce identical results"""
    
    test_cases = [
        (generate_random_prediction_semantics(N), generate_random_prediction_semantics(N)) 
        for _ in range(N_TEST_CASES)
        ]
    
    for i, (semantics_a, semantics_b) in enumerate(test_cases):
        print(f'Test Case {i+1}:', end='')
        
        fast_kemeny = compute_kemeny_distance(semantics_a, semantics_b)
        slow_kemeny = naive_kemeny_distance(semantics_a, semantics_b)
        
        fast_normalised_kemeny = compute_normalised_kemeny_distance(semantics_a, semantics_b)
        slow_normalised_kemeny = naive_normalised_kemeny_distance(semantics_a, semantics_b)
        
        try:
            assert fast_kemeny == slow_kemeny
            assert fast_normalised_kemeny == slow_normalised_kemeny
            print('PASSED')
        except AssertionError:
            print('FAILED!')
            print(f'Efficient Kemeny Result: {fast_kemeny}')
            print(f'Naive Kemeny Result: {slow_kemeny}')
            print(f'Efficient normalised Kemeny Result: {fast_normalised_kemeny}')
            print(f'Naive normalised Kemeny Result: {slow_normalised_kemeny}')
            # Stop if any test case fails
            return
        
    print('All test cases checks passed!')
    
    return test_cases

def run_execution_time():
    
    sizes = list(PERFORMANCE_TEST_SIZES)
    efficient_times = []
    naive_times = []
    
    for n in PERFORMANCE_TEST_SIZES:
        
        # Generate random prediction semantics with length 'n'
        semantics_a = generate_random_prediction_semantics(n)
        semantics_b = generate_random_prediction_semantics(n)
        
        # Efficient algorithm time
        total_time_eff = 0
        for _ in range(TIME_ITERATIONS):
            start_time = time.perf_counter()
            compute_kemeny_distance(semantics_a, semantics_b)
            end_time = time.perf_counter()
            total_time_eff += end_time - start_time
        efficient_times.append(total_time_eff / TIME_ITERATIONS)
        
        # Naive algorithm time
        total_time_naive = 0
        for _ in range(TIME_ITERATIONS):
            start_time = time.perf_counter()
            naive_kemeny_distance(semantics_a, semantics_b)
            end_time = time.perf_counter()
            total_time_naive += end_time - start_time
        naive_times.append(total_time_naive / TIME_ITERATIONS)
        
    return sizes, efficient_times, naive_times

def plot_performance_results(sizes, efficient_times, naive_times):
    """
    Generates a plot comparing the run times of the two algorithms against 
    their theoretical complexities
    """

    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Theoretical O(N log N)
    n_log_n = [n*math.log(n) for n in sizes]
    # Scaling factor to align with the theoretical curve
    scaling_factor_eff = efficient_times[0] / n_log_n[0]
    scaled_n_log_n = [scaling_factor_eff * t for t in n_log_n]
    
    ax.plot(sizes, scaled_n_log_n, label="Theoretical O(N log N)")
    
    # Theoretical O(N^2)
    n_squared = [n**2 for n in sizes]
    # Scaling factor to align with the theoretical curve
    scaling_factor_naive = naive_times[0] / n_squared[0]
    scaled_n_squared = [scaling_factor_naive * t for t in n_squared]
    
    ax.plot(sizes, scaled_n_squared, label="Theoretical O(N^2)")
    
    # Plot observed time as scatter points
    ax.scatter(sizes, efficient_times, color='blue', label="Observed Efficient Time")
    ax.scatter(sizes, naive_times, color='red', label="Observed Naive Time")
        
    ax.set_xlabel("Input Size (N)", fontsize=18)
    ax.set_ylabel("Execution Time (seconds)", fontsize=18)
    ax.set_title("Kemeny Distance Algorithm Performance Comparison", fontsize=22)
    ax.legend(fontsize=16)
    ax.set_yscale("log")  # Use a log scale to better visualise the difference
    
def main():
    
    run_correctness_check()
    sizes, efficient_times, naive_times = run_execution_time()
    plot_performance_results(sizes, efficient_times, naive_times)

if __name__ == "__main__":
    main()