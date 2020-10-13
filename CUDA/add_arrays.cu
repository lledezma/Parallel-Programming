      //adding two arrays and storing the results in a third array using CUDA
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void add(int *a, int *b, int *c, int Num) {
  // int idx = threadIdx.x;             // for a grid with one block of threads
  int idx = threadIdx.x + blockIdx.x * blockDim.x
  if (thread_id < Num){     //comparing bounds
      // c[idx] = a[idx] + b[idx];      // for a grid with one block of threads
      c[idx] = a[idx] + b[idx];
  }
}

int main() {
  int Num = 50;
  int h_a[Num], h_b[Num], h_c[Num]; //declaring host variables
  int *d_a, *d_b, *d_c;             //declaring device variable

  //Memory allocation of device variable
  cudaMalloc((void**)&d_a, Num*sizeof(int));
  cudaMalloc((void**)&d_b, Num*sizeof(int));
  cudaMalloc((void**)&d_c, Num*sizeof(int));

  //filling host array variables
   for(int i = 1; i <= Num; ++i){
      h_a[i-1] = i;
      h_b[i-1] = i;
  }

  //Copy Host memory to Device memory
  cudaMemcpy(d_a,h_a, Num*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_b,h_b, Num*sizeof(int),cudaMemcpyHostToDevice);

  //Launch Kernel 
  // add<<<1,Num>>>(d_a,d_b,d_c,Num);      //A grid with one block and Num=50 threads
  add<<<2,Num/2>>>(d_a,d_b,d_c,Num);       //A grid with two blocks, 50/2 threads per block

  //copy device results to host results
  cudaMemcpy(h_c,d_c, Num*sizeof(int),cudaMemcpyDeviceToHost);

  for(int i = 0;i < Num; i++){  //print results
      printf("%d\n", h_c[i]);
  }

  //free device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  return 0;
}