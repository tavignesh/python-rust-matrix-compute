import time
import random
import numpy as np
from memory_profiler import memory_usage


def generate_matrix(n, m):
    return [[random.random() for _ in range(m)] for _ in range(n)]


def matmul_pure_python(A, B):
    n, m, p = len(A), len(B), len(B[0])
    result = [[0.0 for _ in range(p)] for _ in range(n)]
    for i in range(n):
        for j in range(p):
            for k in range(m):
                result[i][j] += A[i][k] * B[k][j]
    return result


def measure_memory_and_time(func, *args):
    start_time = time.time()
    mem_usage, result = memory_usage((func, args), retval=True, max_usage=True)
    end_time = time.time()
    duration = end_time - start_time
    return duration, mem_usage, result


def benchmark(size):
    print(f"\n🔸 Matrix Size: {size}x{size}")

    # Vanillaadd Python
    A = generate_matrix(size, size)
    B = generate_matrix(size, size)
    if size > 1000:
        print("Pure Python: Time = -NA- s, Peak Mem = -NA- MB")
    else:
        time_py, mem_py, _ = measure_memory_and_time(matmul_pure_python, A, B)
        print(f"Vanilla Python: Time = {time_py:.4f}s, Peak Mem = {mem_py:.2f} MB")

    # NumPy
    time_np, mem_np, _ = measure_memory_and_time(np.dot, A, B)
    print(f"NumPy         : Time = {time_np:.4f}s, Peak Mem = {mem_np:.2f} MB")

if __name__ == "__main__":
    benchmark(int(input("Enter matrix size to generate: ")))
