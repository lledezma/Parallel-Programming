__kernel void addArray(__global char *pattern,                 
                       __global char *str,                    
                       __global int *results,                    
                       int pSize,                 
                       int sSize )                
{                                                            
    //Get our global thread ID                               
    int idx = get_global_id(0);                                 
                                                       
   //Check bounds                                            
   if (idx <= sSize-pSize) 
   {                                                         
       int i;                                                  
       for(i = 0; i < pSize; i++)
       {                                               
           if (pattern[i] != str[idx+i]){            
             break;   
            }                                       
       }                                                      
      if(i == pSize) {
			   atomic_add(results,1); 
		  }                     
    }                                                             
}                                                               