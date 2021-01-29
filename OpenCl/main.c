                //Addind two arrays using all available devices and comparing times
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <OpenCL/opencl.h>
#include <stdio.h>

// The kernel
const char *KernelSource =                                     "\n" \
"__kernel void addArray(  __global int *a,                      \n" \
"                       __global int *b,                        \n" \
"                       __global int *c,                        \n" \
"                       const unsigned int n)                   \n" \
"{                                                              \n" \
"   //Get our global thread ID                                  \n" \
"   int idx = get_global_id(0);                                 \n" \
"                                                               \n" \
"   //Check bounds                                              \n" \
"    while(idx < n){                                            \n" \
"       c[idx] = a[idx] + b[idx];                               \n" \
"       //increment thread index                                \n" \
"       idx += get_num_groups(0) * get_local_size(0);           \n" \
"    }                                                          \n" \
"}                                                              \n" ;

int maxCpuUnits(cl_uint num_platforms, cl_uint platformID, cl_uint num_devices, cl_uint deviceID); //get the max compute units available
const char* deviceName(cl_uint num_platforms, cl_uint platformID, cl_uint num_devices, cl_uint deviceID); //get the name of device

int main(int argc, const char * argv[]) {
    int err; //varible to track errors
    cl_platform_id* platform;
    cl_device_id* device_id;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_event event;
    size_t globalWorkSize;  //global work items
    size_t localWorkSize;   //work items per group

    const int num = 100000000;     //size of arrays
    
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
    
    cl_uint num_platforms; //number of platforms
    err = clGetPlatformIDs(0, NULL, &num_platforms); //get the number of platforms found
    if(err != CL_SUCCESS){
        printf("Error getting the number of platforms\n");
        return EXIT_FAILURE;
    }
    platform = (cl_platform_id*)malloc(sizeof(cl_platform_id)*num_platforms); //allocate memory for platform IDs
    err = clGetPlatformIDs(num_platforms, platform, NULL); //get the IDs of available platforms
    if(err != CL_SUCCESS){
        printf("Error getting the platforms\n");
        return EXIT_FAILURE;
    }
    
    for(cl_uint h = 0; h < num_platforms; h++)
    {

        cl_uint num_devices; // to hold the number of devices found
        err = clGetDeviceIDs(platform[h], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices); //get number of devices found

        device_id = calloc(sizeof(cl_device_id), num_devices); //allocate memory for device IDs
        clGetDeviceIDs(platform[h], CL_DEVICE_TYPE_ALL, num_devices, device_id, NULL); //get ids of all devices available

        //create context
        context = clCreateContext(0, num_devices, device_id, NULL, NULL, &err);
        if(err != CL_SUCCESS){
            printf("Could not create context\n");
            return EXIT_FAILURE;
        }

        for(cl_uint j = 0; j < num_devices; j++)
        {
            // printf("%d\n", j);
            //create command queue
            queue = clCreateCommandQueue(context, device_id[j], CL_QUEUE_PROFILING_ENABLE, &err);
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
            
            globalWorkSize = maxCpuUnits(num_platforms, h, num_devices, j);   //number of global work items
            localWorkSize = 2;                //number of work items per group
            
            //Execute the kernel
            err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, &event);
            if(err != CL_SUCCESS){
                printf("Error executing the Kernel");
                return EXIT_FAILURE;
            }

            //wait for all commands in the queue to be processed and completed
            clWaitForEvents(1, &event);
            clFinish(queue);
            
            //copy device memory to host memory
            err = clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, bytes, h_c, 0, NULL, NULL);
            if(err != CL_SUCCESS){
                printf("Error copying device memory to host memory\n");
                return EXIT_FAILURE;
            }
            
        ///*   print device info, compute units, and results
            printf("Running on device: %s with %d compute units.\n", deviceName(num_platforms, h, num_devices, j), maxCpuUnits(num_platforms, h, num_devices, j));
            // for(int i = 0; i < num; i++){
            //     printf("%d   =   %d  +   %d\n", h_c[i], h_a[i], h_b[i]);
            // }
            cl_ulong time_start;
            cl_ulong time_end;
            clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
            clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
            double nanoSeconds = time_end-time_start;
            printf("OpenCl Execution time is: %0.3f milliseconds \n",nanoSeconds / 1000000.0);
        //*/
        }
    }
    //release device memory
    clReleaseEvent(event);
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

int maxCpuUnits(cl_uint num_platforms, cl_uint platformID, cl_uint num_devices, cl_uint deviceID){
    int err;
    cl_platform_id* platform = (cl_platform_id*)malloc(sizeof(cl_platform_id)*num_platforms); //allocate memory for platform IDs
    err = clGetPlatformIDs(num_platforms, platform, NULL); //get the IDs of available platforms
    if(err != CL_SUCCESS){
        printf("Error getting the platforms\n");
        return EXIT_FAILURE;
    }

    cl_device_id* devices = calloc(sizeof(cl_device_id), num_devices);
    cl_uint maxComputeUnits;
    err = clGetDeviceIDs(platform[platformID], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
    if(err != CL_SUCCESS){
        printf("Error getting device id from maxCpuUnits function\n");
        return EXIT_FAILURE;
    }
    
    err = clGetDeviceInfo(devices[deviceID], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, NULL);
    if(err != CL_SUCCESS){
        printf("Error getting device info from maxCpuUnits function\n");
        return EXIT_FAILURE;
    }
    return maxComputeUnits;
}

const char* deviceName(cl_uint num_platforms, cl_uint platformID, cl_uint num_devices, cl_uint deviceID){
    int err;
    cl_platform_id* platform = (cl_platform_id*)malloc(sizeof(cl_platform_id)*num_platforms); //allocate memory for platform IDs
    err = clGetPlatformIDs(num_platforms, platform, NULL); //get the IDs of available platforms
    if(err != CL_SUCCESS){
        printf("Error getting the platforms\n");
        exit(EXIT_FAILURE);
    }

    cl_device_id* devices = calloc(sizeof(cl_device_id), num_devices);
    err = clGetDeviceIDs(platform[platformID], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
    if(err != CL_SUCCESS){
        printf("Error getting device id from deviceName function\n");
        exit(EXIT_FAILURE);
    }

    size_t valueSize;
    char* nameOfDevice = (char*)malloc(sizeof(valueSize));
    err = clGetDeviceInfo(devices[deviceID], CL_DEVICE_NAME, valueSize, nameOfDevice, NULL);
    if(err != CL_SUCCESS){
        printf("Error getting device name from deviceName function\n");
        exit(EXIT_FAILURE);
    }
    return nameOfDevice;
}

