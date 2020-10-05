      //adding two arrays and storing the results in a third array using CUDA
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void add(int *a, int *b, int *c, int Num) {
  int thread_id = threadIdx.x; 
  if (thread_id < Num){ //compare thread index to make sure we don't go out of bound
      c[thread_id] = a[thread_id] + b[thread_id];
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

  //Launch Kernel (one block with N threads)
  add<<<1,Num>>>(d_a,d_b,d_c,Num);

  //copy device results to host results
  cudaMemcpy(h_c,d_c, Num*sizeof(int),cudaMemcpyDeviceToHost);

  for(int i = 1;i <= Num; i++){  //print results
      printf("%d\n", h_c[i-1]);
  }

  //free device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  return 0;
}