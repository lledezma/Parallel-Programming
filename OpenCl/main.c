                //Array addition using OpenCL
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <OpenCL/opencl.h>
#include <stdio.h>

const char *KernelSource =                                       "\n" \
"__kernel void addArray(  __global int *a,                       \n" \
"                       __global int *b,                         \n" \
"                       __global int *c,                         \n" \
"                       const unsigned int n)                    \n" \
"{                                                               \n" \
"    //Get our global thread ID                                  \n" \
"    int idx = get_global_id(0);                                 \n" \
"                                                                \n" \
"    //Check bounds                                              \n" \
"    if (idx < n)                                                \n" \
"        c[idx] = a[idx] + b[idx];                               \n" \
"}                                                               \n" ;

int cpuUnits(); //get the max compute units available

int main(int argc, const char * argv[]) {
    int err;                    //varible to track errors
    cl_platform_id platform;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    size_t globalWorkSize;  //global work items
    size_t localWorkSize;   //work items per group

    const int num = 12;     //size of arrays
    
    //declaring device variables
    cl_mem d_a;
    cl_mem d_b;
    cl_mem d_c;
    
    //declaring host variables
    int* h_a;
    int* h_b;
    int* h_c;
    
    //initializing host memory
    int bytes = num*sizeof(int);
    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);
    
    for(int i = 0; i < num; i++){
        h_a[i] = i+1;
        h_b[i] = i+1;
    }
    // # of platform IDs || platform || # of OpenCL platforms available
    err = clGetPlatformIDs(1, &platform, NULL);
    if(err != CL_SUCCESS){
        printf("Error getting the platform\n");
        return EXIT_FAILURE;
    }
    
    //get device ID
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    if(err != CL_SUCCESS){
        printf("Error getting the device IDs\n");
        return EXIT_FAILURE;
    }
    
    //create context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if(err != CL_SUCCESS){
        printf("Could not create context\n");
        return EXIT_FAILURE;
    }
    
    //create command queue
    queue = clCreateCommandQueue(context, device_id, 0, &err);
    if(err != CL_SUCCESS){
        printf("Error creating the command queue\n");
        return EXIT_FAILURE;
    }
    
    program = clCreateProgramWithSource(context, 1, (const char **)&KernelSource, NULL, &err);
    if(err != CL_SUCCESS){
        printf("Error creating the program\n");
        return EXIT_FAILURE;
    }
    // built the program
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if(err != CL_SUCCESS){
        printf("Error building the program\n");
        return EXIT_FAILURE;
    }
    
    kernel = clCreateKernel(program, "addArray", &err);
    if(err != CL_SUCCESS){
        printf("error creating the kernel\n");
        return EXIT_FAILURE;
    }
    
    //allocate device memory
    d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &err);
    d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &err);
    d_c = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
    if(err != CL_SUCCESS){
        printf("Error allocating device memory\n");
        return EXIT_FAILURE;
    }
    
    //copy data from host variables to device variables
    err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, bytes, h_a, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0, bytes, h_b, 0, NULL, NULL);
    if(err != CL_SUCCESS){
        printf("Error copying data to device variables\n");
        return EXIT_FAILURE;
    }
    
    // Set the Kernel Arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&d_a);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&d_b);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&d_c);
    err |= clSetKernelArg(kernel, 3, sizeof(int), (void*)&num);
    if(err != CL_SUCCESS){
        printf("Error setting the kernel arguments\n");
        return EXIT_FAILURE;
    }
    
    globalWorkSize = cpuUnits(); //number of global work items
    localWorkSize = 2;           //number of work items per group
                                 //cpuUnits()/2 = # of work groups
    
    //Execute the kernel
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
    if(err != CL_SUCCESS){
        printf("Error executing the Kernel");
        return EXIT_FAILURE;
    }
    
    //wait for all commands in the queue to be processed and completed
    clFinish(queue);
    
    //copy device memory to host memory
    err = clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, bytes, h_c, 0, NULL, NULL);
    if(err != CL_SUCCESS){
        printf("Error copying device memory to host memory\n");
        return EXIT_FAILURE;
    }
    
///*   print device info, compute units, and results
    size_t valueSize;
    char* deviceName = (char*)malloc(sizeof(valueSize));
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, valueSize, deviceName, NULL);
    printf("Running on device: %s with %d compute units.\n", deviceName, cpuUnits());
    for(int i = 0; i < num; i++){
        printf("%d   =   %d  +   %d\n", h_c[i], h_a[i], h_b[i]);
    }
//*/
    //release device memory
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    //release host memory
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;
}

int cpuUnits(){
    int err;
    cl_device_id device_id;
    cl_uint maxComputeUnits;
    err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    if(err != CL_SUCCESS)
        return EXIT_FAILURE;
    
    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, NULL);
    if(err != CL_SUCCESS)
        return EXIT_FAILURE;
    return maxComputeUnits;
}

