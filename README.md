# Matrix Multiplication Benchmark: NumPy vs Rust

This project benchmarks matrix multiplication across four different implementations:
- **Vanilla Python**
- **NumPy** (NumPy isn't Python)
- **Vanilla Rust**
- **Parallel Rust with Rayon**

The focus is on highlighting raw performance, memory usage, and scalability on multi-core systems.

---

## 🔍 Key Observations

- ✅ **NumPy is not really Python**: It leverages compiled C and Fortran libraries (like BLAS/MKL), uses CPU caching, SIMD instructions, and multithreading under the hood.
- ❗ **Vanilla Rust**, despite being compiled, can be slower than NumPy for small matrices due to:
  - Memory access latency
  - Lack of SIMD by default
  - Thread underutilization
  - CPU usage often capped around **10–12%** for single-threaded operations
- 📉 **Memory Efficiency**:
  - **Rust** (both sequential and parallel) uses **minimal memory**, often staying within a few hundred MB even at large matrix sizes.
  - **NumPy and Vanilla Python** can use **2–3 GB of RAM** for large matrix operations due to temporary arrays, caching, and less efficient memory layouts.
  - NumPy’s **reported memory usage** is often misleading, as it doesn’t account for **cached memory** used internally by libraries and the L1, L2, L3 cache abuse.
- 🔥 **Parallel Rust** (via `rayon`) scales beautifully:
  - Utilizes all CPU cores
  - Near-linear speedups on larger matrices (e.g., 1000×1000 and above)
  - Infinitely scalable with cores and threads

---

## 🧪 Tested On

- **CPU**: 8-core Ryzen 7 5800H
- **Matrix Sizes**: 100×100 to 5000×5000
- **Languages**: Python (Basic & NumPy), Rust (Sequential & Parallel)

---

## 📈 Results & Graphs

- Time Taken (Seconds)
  ![image](https://github.com/user-attachments/assets/d8a9005c-4d5d-4062-88de-ab563e7f10a6)<br>
  ![image](https://github.com/user-attachments/assets/24f77949-664b-4f31-bde6-d50aa431bcc4)<br><br>

- RAM Consumed
  ![image](https://github.com/user-attachments/assets/5d54b418-1ea3-457b-a541-a1ce3f7a4152)<br>
  ![image](https://github.com/user-attachments/assets/162c2670-87d6-4b9f-95b5-a737a41d1a45)

---

## 🧠 Conclusion

This project demonstrates that **raw language choice isn't everything**—what matters is how well the code leverages the **hardware**, whether through compiled libraries, parallelism, or memory efficiency. NumPy achieves its performance by using optimized native libraries, while Rust offers **true parallel scalability** with full control and low-level memory efficiency, making it ideal for high-performance computing on modern multi-core systems.
