__kernel void match(__global char *pattern,                 
                    __global char *sequence,                    
                    __global int *results,                    
                             int patLength,                 
                             int seqLength)                
{                                                            
  int idx = get_global_id(0);                       //Get our global thread ID 
  int localCount = 0;                               //local match count 
  int i;
  while(idx <= seqLength-patLength){                //Check bounds
    for(i = 0; i < patLength; i++){
      if(pattern[i] != sequence[idx+i])             //if a mismatch is found, exit the loop. 
        break;
    }
    localCount += (i == patLength) ? 1 : 0;         //if i == patLength, add 1 to localCount
    idx += get_num_groups(0) * get_local_size(0);    //increment our thread index
  }
  atomic_add(results,localCount);     //add the localCount to results (global count)
}


__kernel void match2(__global char *pattern,                 
                     __global char *sequence,                    
                     __global int *results2,                    
                              int patLength,                 
                              int seqLength)                
{                                                            
  int idx = get_global_id(0);                       //Get our global thread ID 
  int localCount = 0;                               //local match count 
  int i = 0;
  while(idx <= seqLength-patLength){
    if(pattern[i] != sequence[idx+i] || i == patLength){
      if(i == patLength) 
        localCount +=1; //if i == patLength, add 1 to localCount
      i = -1; //reset i variable 
      idx += get_num_groups(0) * get_local_size(0); //increment our thread index
    }
    i++;  
  }
  atomic_add(results2,localCount);     //add the localCount to results (global count)
}