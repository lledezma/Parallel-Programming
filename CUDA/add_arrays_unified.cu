		//adding two arrays and storing the results in a third array using CUDA 
							//(Unified Memory Construct)
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void add(int *a, int *b, int *c, int Num) {
	//global thread id
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	//checking bounds
	while(idx < Num) {
		c[idx] = a[idx] + b[idx];
		//increment thread index
  		idx += blockDim.x * gridDim.x;  
	}
}

int main() {
	//size of arrays
	int Num = 100; 
	//size of the arrays
	size_t bytes = Num*sizeof(int);
	//declaring device variables
	int *d_a, *d_b, *d_c;
	
	//Memory allocation for device pointers
	cudaMallocManaged(&d_a, bytes);
	cudaMallocManaged(&d_b, bytes);
	cudaMallocManaged(&d_c, bytes);
	
	//Initializing memory
  	for(int i = 1; i <= Num; ++i){
   	    d_a[i-1] = i;
    	    d_b[i-1] = i;
	}

	//Number a threads per block
	int blockSize = 10;

	//Number of blocks in the grid
	int gridSize = (Num + blockSize - 1) / blockSize;

	//Launch Kernel
	add<<<gridSize,blockSize>>>(d_a,d_b,d_c,Num);

	//wait for all commands to be completed
	cudaDeviceSynchronize();

	//print the results
	for(int i = 0;i < Num; i++){ 
	    printf("%d\n", d_c[0]);
	}

	//Free device memory (Unified)
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}
	
