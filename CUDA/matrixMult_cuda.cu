		//multiplication of two matrices using a kernel with a 2d grid and 2d blocks
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16;

__global__ void multiMatrix(int* A, int* B, int*C, int colA, int colB, int rowA){
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

  	int sum=0;
	if(x < colB && y < rowA){
		for(int i = 0; i < colA; i++){
			sum += A[y * colA + i] * B[i * colB + x];
		}
		C[y * colB + x] = sum;
	}
}

int main(){
	//rows and columns 
	int rowA = 2;
	int colA = 3;
	int rowB = colA;
	int colB = 4;

	//Declaring host variable
	int h_A[colA*rowA], h_B[colB*rowB], h_C[colB*rowA];
	//Declaring device variables
	int *d_A,*d_B,*d_C;

	//Memory allocation of device variables
	cudaMalloc((void**)&d_A, (colA*rowA)*sizeof(int));
	cudaMalloc((void**)&d_B, (colB*rowB)*sizeof(int));
	cudaMalloc((void**)&d_C, (colB*rowA)*sizeof(int));

	//filling host matrices
	for(int i = 0; i < (colA*rowA); i++){
		h_A[i] = i+1;
	}
	for(int i = 0; i < (colB*rowB); i++){
		h_B[i] = i+1;
	}

	//Copy Host memory to Device memory
	cudaMemcpy(d_A,h_A, (colA*rowA)*sizeof(int), cudaMemcpyHostToDevice);
  	cudaMemcpy(d_B,h_B, (colB*rowB)*sizeof(int), cudaMemcpyHostToDevice);

	//Declaring our 2D grid with 2D blocks and dimensions
	unsigned int gridRows = (rowA + BLOCK_SIZE - 1) / BLOCK_SIZE;
	unsigned int gridCols = (colB + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dim3 dimGrid(gridCols, gridRows);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

	//Launch our Kernel
	multiMatrix<<<dimGrid, dimBlock>>>(d_A,d_B,d_C,colA,colB, rowA);

	//copy device results to host 
	cudaMemcpy(h_C,d_C, (colB*rowA)*sizeof(int), cudaMemcpyDeviceToHost);

	//print our results
	for(int i = 0; i < (rowA*colB); i++){
	  printf("%d\n", h_C[i]);
	}

	// free device memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	return 0;
}

