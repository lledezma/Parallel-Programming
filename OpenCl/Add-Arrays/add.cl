// function to find the min value for each thread
int setMin(global int* c, int idx, const unsigned int n)       
{       
    int size = n/get_global_size(0);  
    int min = c[get_global_id(0)*size + 0]; 
    for(int i = 1; i < size; i++)                             
    {     
        idx = get_global_id(0)*size +i;
        if(c[idx] < min)
            min = c[idx];
    }  
    return min;                                              
}                                 

__kernel void addArray(__global int *a,                       
                       __global int *b,                       
                       __global int *c,                       
                       __global int *min_val, const unsigned int n)
{         
   int idx = get_global_id(0);    //thread id                             
   int private_min = 0;           //private min value
   //Check bounds                                             
    while(idx < n){                                           
       c[idx] = a[idx] + b[idx];   //write result to global memory                                                        
       idx += get_num_groups(0) * get_local_size(0);  //increment thread index        
    }   
    mem_fence(CLK_GLOBAL_MEM_FENCE); //all threads have written to global C-array
    if(get_global_id(0) == 0)
        *min_val = c[0];     
    private_min = setMin(c, get_global_id(0), n); 
    if(private_min < *min_val)
        atomic_xchg(min_val, private_min);
} 
