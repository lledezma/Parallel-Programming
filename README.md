# Parallel-Programming

This repository contains parallel programming examples using various APIs and paradigms:

- **CUDA**
- **OpenCL**
- **OpenMP**
- **OpenMPI**
- **POSIX Threads (Pthreads)**

## Directory Structure

- `CUDA/`  
  CUDA programs for array addition, matrix multiplication, and pattern matching.

- `OpenCl/`  
  OpenCL programs for array addition, matrix multiplication, multi-device execution, and pattern matching.

- `OpenMP/`  
  OpenMP programs for matrix multiplication, numerical integration, Monte Carlo simulation, grayscale image conversion, Euler's number calculation, and pattern matching.

- `OpenMPI/`  
  MPI program for parallel pattern matching.

- `Pthreads/`  
  POSIX threads programs for array sum, finding minimum value, and calculating pi.

## Programs

- **Matrix Multiplication:**  
  Implemented in CUDA, OpenCL, and OpenMP.

- **Array Addition:**  
  Implemented in CUDA, OpenCL, and Pthreads.

- **Pattern Match (Naive and Boyer-Moore algorithms):**  
  Implemented in CUDA, OpenCL, OpenMP, and OpenMPI.

- **Finding Minimum Value (Prefix Sum Logic):**  
  Implemented in Pthreads.

- **Euler's Number Calculation:**  
  Implemented in OpenMP.

- **Grayscale Image Conversion:**  
  Implemented in OpenMP.

- **Monte Carlo Method:**  
  Implemented in OpenMP.

- **Numerical Integration (Pi Calculation):**  
  Implemented in OpenMP.

## Experiments and Tests

- OpenCL event profiling for kernel performance and execution time.
- Multithreading with OpenCL to run kernels on multiple devices simultaneously.

## How to Compile and Run

Below are example commands for compiling files using each API. Adjust file names and paths as needed.

### CUDA
```
nvcc -o output_executable source_file.cu
```

### OpenCL
```
gcc -o output_executable source_file.c -lOpenCL
gcc -o output_executable source_file.c -framework opencl
```

### OpenMP
```
gcc -o output_executable source_file.c -fopenmp
```

### OpenMPI
```
mpicc -o output_executable source_file.c
mpirun -np <num_processes> ./output_executable
```

### Pthreads
```
gcc -o output_executable source_file.c -lpthread
```

Refer to the comments in each file for additional compilation flags or requirements. You may need to install CUDA, OpenCL, OpenMP, and MPI libraries to build and run the examples.

## License

This repository is for educational purposes and experimentation with parallel programming techniques.