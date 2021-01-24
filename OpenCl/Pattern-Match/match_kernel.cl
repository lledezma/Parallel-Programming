__kernel void match(__global char *pattern,                 
                       __global char *str,                    
                       __global int *results,                    
                       int pSize,                 
                       int sSize )                
{                                                            
    //Get our global thread ID                               
    int idx = get_global_id(0);                                                       
    //Check bounds                                            
    if(idx <= sSize-pSize){                                                  
        for(int i = 0; i < pSize; i++) {                                      
            if (pattern[i] != str[idx+i]) //if a mismatch is found, exit.        
     			return;               
        }                                     
	    atomic_add(results,1); //a match has been found, so we add 1 to results.                   
    }                                                             
}                                                               

//For number of work items constraint

/*
  int idx = get_global_id(0);                       //Get our global thread ID  
  int i;
  while(idx <= strLength-patLength){                //Check bounds
    for(i = 0; i < patLength; i++){
      if(pattern[i] != str[idx+i])                  //if a mismatch is found, exit. 
        break;
    }
    if(i == patLength)                              //match has been found so we add 1 to results
      atomic_add(results,1);
    idx += get_num_groups(0) * get_local_size(0);    //increment our thread index
  }
*/