    //adding two values using CUDA
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
__global__ void add(int *d_a, int *d_b, int *d_result) {
      *d_result = *d_a + *d_b;
}
int main() {
  //declarre host variables
  int h_a = 50;
  int h_b = 50;
  int h_result = 0;
  
  //declare device variables
  int *d_a;
  int *d_b;
  int *d_result;

  //Memory allocation of device variable
  cudaMalloc((void**)&d_a, sizeof(int));
  cudaMalloc((void**)&d_b, sizeof(int));
  cudaMalloc((void**)&d_result, sizeof(int));

  //Copy Host memory to Device memory
  cudaMemcpy(d_a, &h_a, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, &h_b, sizeof(int), cudaMemcpyHostToDevice);

  //Launch Kernel
  add<<<1,1>>>(d_a,d_b,d_result);

  //copy device results to host results
  cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

  printf("the result is: %d\n",h_result);

  //free device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_result);
  return 0;
}