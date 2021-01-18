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
       int i;                                                  
       for(i = 0; i < pSize; i++) {         
            //if a letter mismatch is found, exit.                                      
           if (pattern[i] != str[idx+i])         
             return;               
       }
       //if i == size, that means a match has been found.                                                    
	     atomic_add(results,1); //add one to results                    
    }                                                             
}                                                               
