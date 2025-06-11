import time
import random
import numpy as np

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

def benchmark(size):
    print(f"\nMatrix Size: {size}x{size}")

    # Pure Python
    A = generate_matrix(size, size)
    B = generate_matrix(size, size)
    start_py = time.time()
    matmul_pure_python(A, B)
    end_py = time.time()
    print(f"Pure Python Time: {end_py - start_py:.4f} seconds")

    # NumPy
    start_np = time.time()
    np.dot(A, B)
    end_np = time.time()
    print(f"NumPy Time      : {end_np - start_np:.4f} seconds")

if __name__ == "__main__":
    benchmark(int(input("Enter matrix size to generate: ")))
