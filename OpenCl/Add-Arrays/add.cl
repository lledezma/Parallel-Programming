// function to the min value for each thread
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

// old , cmp, val,                  (old == cmp) ? val : old
#define LOCK(lock) atomic_cmpxchg(lock, 0, 1)
#define UNLOCK(lock) atomic_xchg(lock, 0)

__kernel void addArray(__global int *a,                       
                       __global int *b,                       
                       __global int *c,                       
                       __global int *min_val, const unsigned int n)
{         
   //Get our global thread ID  
   int idx = get_global_id(0);                                
   int private_min = 0;  
   __global int* mutex;         //global mutex lock
   __local int group_min_val;   //local/group min value;
   //Check bounds                                             
    while(idx < n){                                           
       c[idx] = a[idx] + b[idx];                              
       //increment thread index                               
       idx += get_num_groups(0) * get_local_size(0);  //increment thread index        
    }   
    barrier(CLK_GLOBAL_MEM_FENCE); //all threads have written to global C-array
    group_min_val = c[n-1];     //can be any value in the c array
    *min_val = c[0];            //can be any value in the c array
    private_min = setMin(c, get_global_id(0), n);  
    while(LOCK(mutex)) //lock
    {
        if(private_min < group_min_val)
            group_min_val = private_min;
        UNLOCK(mutex);  //unlock
    }
    barrier(CLK_GLOBAL_MEM_FENCE); //sync all global threads
    if(get_local_id(0) == 0) //threads with local id = 0 write to global min
    {
        while(LOCK(mutex))
        {
            if(group_min_val < *min_val)
                atomic_xchg(min_val, group_min_val);
            UNLOCK(mutex);
        }
    }
}                                                          