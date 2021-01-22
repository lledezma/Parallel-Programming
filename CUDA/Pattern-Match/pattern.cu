			//Pattern Match Program using CUDA 
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void match(char* pattern, char* string, int* results, int patLength, int strLength){
	//get global thread id
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  //check bounds
  if (x <= strLength-patLength){                                                         
    for(int i = 0; i < patLength; i++){                                               
      if (pattern[i] != string[idx+i])   //if a mismatch is found, exit.        
        return;                                    
    }
    atomicAdd(results,1); //match has been found so we add 1 to results
  }       
}

int main(){
  //device variables
  char* d_pattern;       //device pattern
  char* d_string;        //device string
  int* d_results;        //device variable to store results

  //host variables
  int h_results;        //host varible to store results  
  const char* h_pattern = "NNTHVLTLP";
  const char* h_string = "MIVNNTHVLTLPLYTTTTCHTHPHLYTNNTHVLTLPYSIYHLKLTLLSDSTSLHGPSCHTHNNTHVLTLPTHVLTLLTLLSDSTSRWGSK";
  int h_patLength = (int)strlen(h_pattern); //length of pattern
  int h_strLength = (int)strlen(h_string);  //length of string

  //Memory allocation of device variables
  cudaMalloc((void**)&d_pattern, h_patLength*sizeof(char));
  cudaMalloc((void**)&d_string, h_strLength*sizeof(char));
  cudaMalloc((void**)&d_results, sizeof(int));

  //Copy Host memory to Device Memory
  cudaMemcpy(d_pattern,h_pattern, h_patLength*sizeof(char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_string,h_string, h_strLength*sizeof(char), cudaMemcpyHostToDevice);

  //Launch Kernel
  match<<<1,h_strLength>>>(d_pattern,d_string,d_results,h_patLength, h_strLength);      //A grid with one block and strLength threads.

  //copy device results to host results
  cudaMemcpy(&h_results, d_results, sizeof(int), cudaMemcpyDeviceToHost);

  //print results 
  printf("Total number of matches: %d\n", h_results);

  //free device memory
  cudaFree(d_pattern);
  cudaFree(d_string);
  cudaFree(d_results);

  return 0;
}
