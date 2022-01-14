			//Pattern Match Program using CUDA 
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void match(char* pattern, char* sequence, int* results, int patLength, int seqLength){
	//get global thread id
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

  //check bounds
  if (idx <= seqLength-patLength){                                                         
    for(int i = 0; i < patLength; i++){                                               
      if (pattern[i] != sequence[idx+i])   //if a mismatch is found, exit.        
        return;                                    
    }
    atomicAdd(results,1); //match has been found so we add 1 to results
  }       
}

int main(){
  //device variables
  char* d_pattern;       //device pattern
  char* d_sequence;      //device sequence
  int* d_results;        //device variable to store results

  //host variables
  int h_results;        //host varible to store results  
  const char* h_pattern = "GGATCGA";
  const char* h_sequence = "GAATTGAATTCAGGATCGAGTTACAGTTAAATTCAGTTACGGATCGAAGTTA\n\
                            AGTTAAGTTAGAATATTCAGTGGATCGATACAGTTAAATTCAGTTACACAGT\n\
                            TGGATCGAAAGTTAAGTTAGAATATTCAGTTAGGAATTCAGGGATCGATTAC\n\
                            AGTTAAATTCAGTTTTAAGTTAATCAGTTAC";
  int h_patLength = (int)strlen(h_pattern); //length of pattern
  int h_seqLength = (int)strlen(h_sequence);  //length of sequence

  //Memory allocation of device variables
  cudaMalloc((void**)&d_pattern, h_patLength*sizeof(char));
  cudaMalloc((void**)&d_sequence, h_seqLength*sizeof(char));
  cudaMalloc((void**)&d_results, sizeof(int));

  //Copy Host memory to Device Memory
  cudaMemcpy(d_pattern,h_pattern, h_patLength*sizeof(char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_sequence,h_sequence, h_seqLength*sizeof(char), cudaMemcpyHostToDevice);

  //Launch Kernel
  match<<<1,h_seqLength>>>(d_pattern,d_sequence,d_results,h_patLength, h_seqLength); //A grid with one block and strLength threads.

  //copy device results to host results
  cudaMemcpy(&h_results, d_results, sizeof(int), cudaMemcpyDeviceToHost);

  //print results 
  printf("Total number of matches: %d\n", h_results);

  //free device memory
  cudaFree(d_pattern);
  cudaFree(d_sequence);
  cudaFree(d_results);

  return 0;
}
