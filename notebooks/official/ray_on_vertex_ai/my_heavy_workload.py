import ray
import time

# Initialize Ray
ray.init()

# Define a computationally intensive task
@ray.remote(num_cpus=1)
def heavy_task(x):
    """
    Simulates a heavy workload by performing a CPU-bound operation.
    This example calculates the sum of squares for a range of numbers.
    """
    total = 0
    for i in range(x):
        total += i * i
    time.sleep(1)  # Simulate some work duration
    return total

# Generate a large number of tasks
num_tasks = 1000
results = []
for i in range(num_tasks):
    results.append(heavy_task.remote(1000000))

# Retrieve results (this will trigger autoscaling if needed)
outputs = ray.get(results)

# Print the sum of the results (optional)
print(f"Sum of results: {sum(outputs)}")

# Terminate the process
ray.shutdown()
