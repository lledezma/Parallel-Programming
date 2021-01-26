__kernel void match(__global char *pattern,                 
                       __global char *str,                    
                       __global int *results,                    
                       int patLength,                 
                       int strLength)                
{                                                            
  int localCount = 0;                               //local match count
  int idx = get_global_id(0);                       //Get our global thread ID  
  int i;
  while(idx <= strLength-patLength){                //Check bounds
    for(i = 0; i < patLength; i++){
      if(pattern[i] != str[idx+i])                  //if a mismatch is found, exit the loop. 
        break;
    }
    if(i == patLength)                              //match has been found so we add 1 to localCount
      localCount+=1;
    idx += get_num_groups(0) * get_local_size(0);    //increment our thread index
  }
  atomic_add(results,localCount);     //add the localCount to results (global count)
}
