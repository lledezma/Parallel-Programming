#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <OpenCL/opencl.h>

const int num = 100000000;

// The kernel
const char *KernelSource =                                     "\n"\
"__kernel void addArray(__global int *a,                        \n"\
"                       __global int *b,                        \n"\
"                       __global int *c,                        \n"\
"                       const unsigned int n)                   \n"\
"{                                                              \n"\
"   //Get our global thread ID                                  \n"\
"   int idx = get_global_id(0);                                 \n"\
"                                                               \n"\
"   //Check bounds                                              \n"\
"    while(idx < n){                                            \n"\
"       c[idx] = a[idx] + b[idx];                               \n"\
"       //increment thread index                                \n"\
"       idx += get_num_groups(0) * get_local_size(0);           \n"\
"    }                                                          \n"\
"}                                                              \n";


struct DataStruct {
  int thread_id;
  cl_device_id device;
  cl_context context;
  int* h_a;
  int* h_b;
  int* h_c;
};


void *routine(void* structData)
{
  struct DataStruct* data = (struct DataStruct*)structData;
  cl_uint err;             // var to track errors
  cl_command_queue queue;
  cl_program program;
  cl_kernel kernel;
  cl_event event;
  cl_ulong time_start;        //event start time
  cl_ulong time_end;          //event end time
  cl_uint maxComputeUnits;    //max compute units
  cl_device_type device_type; //device type
  char nameOfDevice[128];     //name of device
  size_t maxWorkItems;        //max work group size
  size_t globalWorkSize;      //global work items
  size_t localWorkSize;       //work items per group
  
  //allocating host memory
  data->h_c = (int*)malloc(num*sizeof(int));

  //declaring device variables
  cl_mem d_a;
  cl_mem d_b;
  cl_mem d_c;

  //get the device type
  err = clGetDeviceInfo(data->device, CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, NULL);
  if(err != CL_SUCCESS){
      printf("Error getting the device type.\n");
      exit(EXIT_FAILURE);
  }

  //get the max compute units
  err = clGetDeviceInfo(data->device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, NULL);
  if(err != CL_SUCCESS){
      printf("Error getting the max compute units.\n");
      exit(EXIT_FAILURE);
  }

  //get the max work group size
  err = clGetDeviceInfo(data->device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkItems), &maxWorkItems, NULL);
  if(err != CL_SUCCESS){
      printf("Error getting device info from maxCompUnits function.\n");
      exit(EXIT_FAILURE);
  }

  //get the name of the device
  err = clGetDeviceInfo(data->device, CL_DEVICE_NAME, 128, nameOfDevice, NULL);
  if(err != CL_SUCCESS){
      printf("Error getting device name of the device.\n");
      exit(EXIT_FAILURE);
  }

  //create command queue
  queue = clCreateCommandQueue(data->context, data->device, CL_QUEUE_PROFILING_ENABLE, &err);
  if(err != CL_SUCCESS){
      printf("Error creating the command queue.\n");
      exit(EXIT_FAILURE);
  }

  //create the program
  program = clCreateProgramWithSource(data->context, 1, (const char **)&KernelSource, NULL, &err);
  if(err != CL_SUCCESS){
      printf("Error creating the program.\n");
      exit(EXIT_FAILURE);
  }

  // build the program
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if(err != CL_SUCCESS){
      printf("Error building the program.\n");
      exit(EXIT_FAILURE);
  }

  //create the kernel
  kernel = clCreateKernel(program, "addArray", &err);
  if(err != CL_SUCCESS){
      printf("error creating the kernel.\n");
      exit(EXIT_FAILURE);
  }

  //allocate device memory
  d_a = clCreateBuffer(data->context, CL_MEM_READ_ONLY, num*sizeof(int), NULL, &err);
  d_b = clCreateBuffer(data->context, CL_MEM_READ_ONLY, num*sizeof(int), NULL, &err);
  d_c = clCreateBuffer(data->context, CL_MEM_READ_WRITE, num*sizeof(int), NULL, &err);
  if(err != CL_SUCCESS){
      printf("Error allocating device memory.\n");
      exit(EXIT_FAILURE);
  }

  //copy data from host variables to device variables
  err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, num*sizeof(int), data->h_a, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0, num*sizeof(int), data->h_b, 0, NULL, NULL);
  if(err != CL_SUCCESS){
      printf("Error copying data to device variables.\n");
      exit(EXIT_FAILURE);
  }

  // Set the Kernel Arguments
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&d_a);
  err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&d_b);
  err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&d_c);
  err |= clSetKernelArg(kernel, 3, sizeof(int), (void*)&num);
  if(err != CL_SUCCESS){
      printf("Error setting the kernel arguments.\n");
      exit(EXIT_FAILURE);
  }

  if(device_type == CL_DEVICE_TYPE_CPU){
    globalWorkSize = 6;
    localWorkSize = 1;
  }
  else{
    globalWorkSize = maxComputeUnits*maxWorkItems;
    localWorkSize = maxWorkItems;
  }

  //Execute the kernel
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, &event);
  if(err != CL_SUCCESS){
      printf("Error executing the Kernel.\n");
      exit(EXIT_FAILURE);
  }

  //wait for all commands in the queue to be processed and completed
  clWaitForEvents(1, &event);
  clFinish(queue);

  //copy device memory to host memory
  err = clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, num*sizeof(int), data->h_c, 0, NULL, NULL);
  if(err != CL_SUCCESS){
      printf("Error copying device memory to host memory.\n");
      exit(EXIT_FAILURE);
  }
  
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
  double nanoSeconds = time_end-time_start;
  //*   print device info, compute units, processing elements and results
  printf("Thread id: %d\n", data->thread_id);
  printf("Running on device: %s with %d compute units.\n", nameOfDevice, maxComputeUnits);
  printf("OpenCl Execution time is: %0.3f milliseconds. \n\n",nanoSeconds / 1000000.0);

  //release device memory
  clReleaseEvent(event);
  clReleaseMemObject(d_a);
  clReleaseMemObject(d_b);
  clReleaseMemObject(d_c);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);

}

int main()
{
  int err;
  cl_platform_id platform;
  cl_device_id* device_ids; //device ids
  cl_context context;
  cl_uint num_devices;      //number of devices found

  //declaring host variables
  int* h_a;
  int* h_b;

  //initializing host memory
  size_t bytes = num*sizeof(int);
  h_a = (int*)malloc(bytes);
  h_b = (int*)malloc(bytes);

  for(int i = 0; i < num; i++){
      h_a[i] = i+1;
      h_b[i] = i+1;
  }

  //get plaform id
  err = clGetPlatformIDs(1, &platform, NULL);
  if(err != CL_SUCCESS){
      printf("Error getting the platforms.\n");
      return EXIT_FAILURE;
  }

  //get number of devices found
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
  if(err != CL_SUCCESS){
      printf("Error getting the number of devices found.\n");
      return EXIT_FAILURE;
  }

  //allocate memory for device IDs
  device_ids = calloc(sizeof(cl_device_id), num_devices);

  //get ids of all available devices
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, device_ids, NULL);
  if(err != CL_SUCCESS){
      printf("Error getting IDs of devices.\n");
      return EXIT_FAILURE;
  }

  //create context
  context = clCreateContext(0, num_devices, device_ids, NULL, NULL, &err);
  if(err != CL_SUCCESS){
      printf("Could not create context.\n");
      return EXIT_FAILURE;
  }

  pthread_t threads[num_devices];
  struct DataStruct thread_structs[num_devices];

  for(cl_uint device_id = 0; device_id < num_devices; device_id++)
  {
    thread_structs[device_id].thread_id = device_id;
    thread_structs[device_id].device = device_ids[device_id];
    thread_structs[device_id].context = context;
    thread_structs[device_id].h_a = h_a;
    thread_structs[device_id].h_b = h_b;
    pthread_create(&threads[(long)device_id], NULL, routine, (void*) &thread_structs[device_id]);
  }

  for(long i = 0; i < num_devices; i++){
    pthread_join(threads[i], NULL);
  }

  //release host memory
  for(int i = 0; i < num_devices; i++){
    free(thread_structs[i].h_c);
  }
  free(h_a);
  free(h_b);
  free(device_ids);

  //release device memory
  clReleaseContext(context);
  return 0;
}


