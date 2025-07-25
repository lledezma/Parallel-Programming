                //Matrix multiplication using OpenCL
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <OpenCL/opencl.h>

const char *KernelSource =                                    "\n"
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable                 \n"
"kernel void multiMatrix(__global int *A,                      \n"
"                        __global int *B,                      \n"
"                        __global int *C,                      \n"
"                        int colA, int colB, int rowA)         \n"
"{                                                             \n"
"    //Get our global thread ID                                \n"
"    int x = get_global_id(0);                                 \n"
"    int y = get_global_id(1);                                 \n"
"                                                              \n"
"    //Check bounds                                            \n"
"    int sum = 0;                                              \n"
"    if(x < colB && y < rowA){                                 \n"
"       for(int i = 0; i < colA; ++i){                         \n"
"           sum += A[y * colA + i] * B[i * colB + x];          \n"
"       }                                                      \n"
"        C[y * colB + x] = sum;                                \n"
"     }                                                        \n"
"}                                                             \n";

int main(int argc, const char * argv[]) {
    cl_int err;                    //varible to track errors
    cl_platform_id platform;
    cl_device_id device_id;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    
    //2d NDRange and workgroup
    size_t globalWorkSize[2] = {16, 16};  //NDRange (global work items)
    size_t localWorkSize[2] = {8, 8};   //workgroup (local work items)
    
    //declaring device variables
    cl_mem d_a;
    cl_mem d_b;
    cl_mem d_c;
    
    //declaring host variables
    int* h_a;
    int* h_b;
    int* h_c;
    
    //rows and columns
    const int rowA = 15;
    const int colA = 15;
    const int rowB = colA;
    const int colB = 10;
    
    //allocating host memory
    h_a = (int*)malloc((colA*rowA)*sizeof(int));
    h_b = (int*)malloc((colB*rowB)*sizeof(int));
    h_c = (int*)malloc((colB*rowA)*sizeof(int));
    
    //initializing host variables (Matrices)
    for(int i = 0; i < (colA*rowA); i++){
        h_a[i] = i+1;
    }

    for(int i = 0; i < (colB*rowB); i++){
        h_b[i] = i+1;
    }

    // # of platform IDs || platform || # of OpenCL platforms available
    err = clGetPlatformIDs(1, &platform, NULL);
    if(err != CL_SUCCESS){
        printf("Error getting the platform\n");
        return EXIT_FAILURE;
    }
    
    //get device ID
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    if(err != CL_SUCCESS){
        printf("Error getting the device IDs\n");
        return EXIT_FAILURE;
    }

    //get the name of the device
    char nameOfDevice[128];
    err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, 128, nameOfDevice, NULL);
    if(err != CL_SUCCESS){
        printf("Error getting the name of the device\n");
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
    
    //create the program
    program = clCreateProgramWithSource(context, 1, (const char **)&KernelSource, NULL, &err);
    if(err != CL_SUCCESS){
        printf("Error creating the program\n");
        return EXIT_FAILURE;
    }

    // build the program
    err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if(err != CL_SUCCESS){
        printf("Error building the program\n");
        return EXIT_FAILURE;
    }
    
    //create the kernel
    kernel = clCreateKernel(program, "multiMatrix", &err);
    if(err != CL_SUCCESS){
        printf("error creating the kernel\n");
        return EXIT_FAILURE;
    }
    
    //allocating device memory
    d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, (colA*rowA)*sizeof(int), NULL, &err);
    d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, (colB*rowB)*sizeof(int), NULL, &err);
    d_c = clCreateBuffer(context, CL_MEM_READ_WRITE, (colB*rowA)*sizeof(int), NULL, &err);
    if(err != CL_SUCCESS){
        printf("Error allocating device memory\n");
        return EXIT_FAILURE;
    }
    
    //copy data from host variables to device variables
    err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, (colA*rowA)*sizeof(int), h_a, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0, (colB*rowB)*sizeof(int), h_b, 0, NULL, NULL);
    if(err != CL_SUCCESS){
        printf("Error copying data to device variables\n");
        return EXIT_FAILURE;
    }
    
    // Set the Kernel Arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&d_a);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&d_b);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&d_c);
    err |= clSetKernelArg(kernel, 3, sizeof(int), (void*)&colA);
    err |= clSetKernelArg(kernel, 4, sizeof(int), (void*)&colB);
    err |= clSetKernelArg(kernel, 5, sizeof(int), (void*)&rowA);
    if(err != CL_SUCCESS){
        printf("Error setting the kernel arguments\n");
        return EXIT_FAILURE;
    }

    //Execute the kernel
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    if(err != CL_SUCCESS){
        printf("Error executing the Kernel");
        return EXIT_FAILURE;
    }
    
    //wait for all commands in the queue to be processed and completed
    clFinish(queue);
    
    //copy device memory to host memory
    err = clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, (colB*rowA)*sizeof(int), h_c, 0, NULL, NULL);
    if(err != CL_SUCCESS){
        printf("Error copying device memory to host memory\n");
        return EXIT_FAILURE;
    }
    
///*   print result
    for(int i = 0; i < colB*rowA; i++){
        printf("%d ", h_c[i]);
	if(((i+1) % colB) == 0)
		printf("\n");
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

