                //Pattern-Match Program using OpenCL
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <OpenCL/opencl.h>
#include <stdio.h>

long LoadOpenFile(char const* path, char **buf); //Function to Open Files
int MaxCompUnits(); //get the max compute units available
const char* deviceName(); //get the name of device

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
    
    //device variables
    cl_mem d_pattern;       //device pattern
    cl_mem d_string;        //device sequence
    cl_mem d_results;       //device variable to store results
    
    //host variables
    int h_results;          //host variable to store results
    char* h_pattern = "NNTHVLTLP";      
    char* h_string = "MIVNNTHVLTLPLYTTTTCHTHPHLYTNNTHVLTLPYSIYHLKLTLLNNTHVLTLPSDSTSLHGPSCHTHNNTHVLTLPTHVLTLLTLLSDSTSRWGSKNNTHVLTLP";
    int h_pSize = (int)strlen(h_pattern);       //length of pattern
    int h_sSize = (int)strlen(h_string);        //length of sequence

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
    
    // Create the compute program from the source file
   char *KernelSource;
   long lFileSize;
   lFileSize = LoadOpenFile("match_kernel.cl", &KernelSource);
   if( lFileSize < 0L ) {
       perror("File read failed");
       return 1;
   }

    program = clCreateProgramWithSource(context, 1, (const char **)&KernelSource, NULL, &err);
    if(err != CL_SUCCESS){
        printf("Error creating the program\n");
        return EXIT_FAILURE;
    }

    // built the program
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if(err != CL_SUCCESS){
        printf("Error building the program.....%d\n",err);
        return EXIT_FAILURE;
    }

    kernel = clCreateKernel(program, "match", &err);
    if(err != CL_SUCCESS){
        printf("error creating the kernel\n");
        return EXIT_FAILURE;
    }
    
    //allocate device memory and copy host memory to device memory
    d_pattern = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, h_pSize, h_pattern, &err);
    d_string = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, h_sSize, h_string, &err);
    d_results = clCreateBuffer(context, CL_MEM_READ_WRITE, 1, NULL, &err);
    if(err != CL_SUCCESS){
        printf("Error allocating device memory\n");
        return EXIT_FAILURE;
    }

    // Set the Kernel Arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&d_pattern);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&d_string);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&d_results);
    err |= clSetKernelArg(kernel, 3, sizeof(int), (void*)&h_pSize);
    err |= clSetKernelArg(kernel, 4, sizeof(int), (void*)&h_sSize);
    if(err != CL_SUCCESS){
        printf("Error setting the kernel arguments\n");
        return EXIT_FAILURE;
    }
    
    globalWorkSize = MaxCompUnits();   //number of global work items
    localWorkSize = globalWorkSize/4;  //number of work items per group

    //Execute the kernel
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
    if(err != CL_SUCCESS){
        printf("Error executing the Kernel\n");
        return EXIT_FAILURE;
    }

    //wait for all commands in the queue to be processed and completed
    clFinish(queue);

    //copy device memory to host memory
    err = clEnqueueReadBuffer(queue, d_results, CL_TRUE, 0, 1, &h_results, 0, NULL, NULL);
    if(err != CL_SUCCESS){
        printf("Error copying device memory to host memory\n");
        return EXIT_FAILURE;
    }
    
    //print the results. The total number of matches. 
    printf("Total number of matches: %d\n", h_results);

    //release device memory
    clReleaseMemObject(d_pattern);
    clReleaseMemObject(d_string);
    clReleaseMemObject(d_results);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;
}


long LoadOpenFile(char const* path, char **buf){
    FILE  *fp;
    size_t fsz;
    long   off_end;
    int    rc;

    /* Open the file */
    fp = fopen(path, "r");
    if( NULL == fp ) {
        return -1L;
    }
    rc = fseek(fp, 0L, SEEK_END);
    if( 0 != rc ) {
        return -1L;
    }
    if( 0 > (off_end = ftell(fp)) ) {
        return -1L;
    }
    fsz = (size_t)off_end;
    *buf = (char *) malloc( fsz+1);
    if( NULL == *buf ) {
        return -1L;
    }
    rewind(fp);
    if( fsz != fread(*buf, 1, fsz, fp) ) {
        free(*buf);
        return -1L;
    }
    if( EOF == fclose(fp) ) {
        free(*buf);
        return -1L;
    }
    (*buf)[fsz] = '\0';
    return (long)fsz;
}

int MaxCompUnits(){
    int err;
    cl_device_id device_id;
    cl_uint maxComputeUnits;
    err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    if(err != CL_SUCCESS){
        printf("Error getting device id from maxCompUnits function\n");
        return EXIT_FAILURE;
    }
    
    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, NULL);
    if(err != CL_SUCCESS){
        printf("Error getting device info from maxCompUnits function\n");
        return EXIT_FAILURE;
    }
    return maxComputeUnits;
}

const char* deviceName(){
    int err;
    cl_device_id device_id;
    err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    if(err != CL_SUCCESS){
        printf("Error getting device id from deviceName function\n");
        exit(EXIT_FAILURE);
    }

    char* nameOfDevice = (char*)malloc(sizeof(char)*128);
    err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(char)*128, nameOfDevice, NULL);
    if(err != CL_SUCCESS){
        printf("Error getting device name from deviceName function\n");
        exit(EXIT_FAILURE);
    }
    return nameOfDevice;
}

