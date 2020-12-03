#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
			//Global variables
int numberofthreads=4;
int sum=0;
#define length 10000
int myArray[length];
pthread_mutex_t mutex1;

//Method to add array values
void *addMethod(void* threadid);

int main() {
	long i;
	pthread_t thread[numberofthreads];
	pthread_mutex_init(&mutex1,NULL);

	//Read data array length from a file;
	FILE *myFile;

	if((myFile = fopen("file.txt","r")) == NULL) {
		printf("No such file\n");
		exit(1);
	}
	for(i=1; i <= numberofthreads; i++) {
		fscanf(myFile, "%d", &myArray[i]);
		printf("%d\n",myArray[i]);
	}
	for(i=1; i <= numberofthreads; i++) {
		pthread_create(&thread[i], NULL, addMethod, (void*) i);
	}
	printf("Hello from the main thread\n");
	for(i=1; i <= numberofthreads; i++){
		pthread_join(thread[i], NULL);
	}
	printf("The sum is %d\n", sum);
	return 0;
}

void *addMethod(void* threadid){
	int i, local_sum =0;
	printf("Hello from the threadid: %ld\n", (long)threadid);
	for(i= ((long)threadid-1)*length/numberofthreads; i < (long)threadid*length/numberofthreads; i++)
	{
		local_sum+=myArray[i];
	}
	pthread_mutex_lock(&mutex1);
	sum+=local_sum;
	pthread_mutex_unlock(&mutex1);
	return NULL;
}
