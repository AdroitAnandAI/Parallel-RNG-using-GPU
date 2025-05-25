# ⚡ High-Performance Parallel Random Number Generators on CUDA: ALFG & GFSR Algorithms

## Summary

This project demonstrates the **parallelization of inherently sequential pseudorandom number generators (PRNGs)** using mathematical hacks in **NVIDIA CUDA**. 

It includes:

- **Additive Lagged Fibonacci Generator (ALFG)**  
  Implemented using the *Continuous Subsequence Technique* to enable efficient parallel execution across GPU threads.

- **Generalized Feedback Shift Register (GFSR)**  
  Employs the *Leapfrog Technique* for distributing computation, significantly enhancing parallel performance.

These CUDA-based implementations leverage the **massive parallelism** of GPU architecture to achieve high-throughput random number generation, making them ideal for:

- Monte Carlo simulations  
- Stochastic modeling  
- Scientific computing applications  


## 🚀 Main CUDA Source Codes

The following are the primary CUDA source files in this project:

| File Name        | Description                 |
|------------------|-----------------------------|
| `ALFG_Multi.cu`  | Multi-threaded ALFG logic   |
| `ALFG_Single.cu` | Single-threaded ALFG logic  |
| `GFSR.cu`        | GFSR random number generator |

> 📝 These files implement GPU-based algorithms for high-performance number generation and simulation.


Usage:

## 🧪 Compile and Run CUDA Program (`ALFG_Multi.cu`)

### Step 1: Compile with `nvcc`

```bash
nvcc -o ALFG_Multi ALFG_Multi.cu
```
### Step 2: Run the Executable
```
./ALFG_Multi
```

You can compile and run the *.cu files in the same manner.

# 🔍 Comparison of CUDA-Based Pseudorandom Number Generators

This project implements three distinct pseudorandom number generators (PRNGs) using NVIDIA CUDA:

- **ALFG_Single.cu**: Single-threaded Additive Lagged Fibonacci Generator
- **ALFG_Multi.cu**: Multi-threaded Additive Lagged Fibonacci Generator
- **GFSR.cu**: Generalized Feedback Shift Register Generator

Each implementation showcases different strategies for parallelization and performance optimization on GPU architectures.

---

## 🧵 `ALFG_Single.cu`: Single-Threaded Execution

- **Execution Model**: Utilizes a single GPU thread to generate the entire sequence of random numbers.
- **Implementation**:
  - The kernel function is launched with a single thread.
  - A loop within the kernel computes the ALFG sequence serially.
  - The generated numbers are stored in global memory.
- **Use Case**: Suitable for testing, debugging, or scenarios where parallelization overhead outweighs performance gains.

---

## 🌐 `ALFG_Multi.cu`: Multi-Threaded Parallel Execution

- **Execution Model**: Leverages multiple GPU threads to generate random numbers in parallel.
- **Implementation**:
  - The kernel function is launched with multiple threads, each responsible for computing a portion of the ALFG sequence.
  - Each thread calculates its indices based on thread and block IDs to avoid overlap.
  - Synchronization mechanisms are used when necessary.
- **Use Case**: Ideal for high-performance applications requiring large volumes of random numbers (e.g., Monte Carlo simulations).

---

## 🔁 `GFSR.cu`: Generalized Feedback Shift Register Implementation

- **Execution Model**: Employs parallel GPU threads to implement the GFSR pseudorandom number algorithm.
- **Implementation**:
  - Each thread computes a segment of the sequence using feedback from previous values.
  - The GFSR algorithm uses XOR operations and lagged feedback shifts.
  - Threads operate independently to maximize throughput.
- **Use Case**: Well-suited for fast, high-quality random number generation in massively parallel settings.

---

## ⚖️ Summary of Differences

| Feature              | `ALFG_Single.cu`               | `ALFG_Multi.cu`                        | `GFSR.cu`                              |
|----------------------|--------------------------------|----------------------------------------|----------------------------------------|
| Execution Model      | Single-threaded                | Multi-threaded                         | Multi-threaded                         |
| Parallelization      | None                           | Yes                                    | Yes                                    |
| Performance          | Lower                          | High                                   | High                                   |
| Implementation       | Simple loop in single kernel   | Thread-indexed computation             | Parallel shift-register based          |
| Ideal Use Case       | Testing, debugging             | Monte Carlo, simulations               | Fast PRNG in large-scale parallel tasks|

---

In summary:

- `ALFG_Single.cu` provides a basic serial approach to understand the ALFG logic.
- `ALFG_Multi.cu` demonstrates effective use of GPU threads for scalable PRNG generation.
- `GFSR.cu` introduces an efficient, parallel, and XOR-based approach using the GFSR technique.

Each implementation reflects different levels of parallelism and performance tuning for CUDA-capable devices.
