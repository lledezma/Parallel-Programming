			//Adding two arrays and finding the min value. Using prefix sum algorithm logic
			//to find min value. Multithreading program using POSIX Threads
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

//Global variables and Shared variables

pthread_mutex_t pmutex;	//mutex lock
int  a[100];
int  b[100];
int  c[100];	//array to store the sums of a and b arrays
int num_of_threads = 4;	//number of threads
int n = sizeof(a)/sizeof(a[0]);	//size of arrays

void *mySum(void* threadid) ; //function to add our arrays
void *minVal(void* threadid); //function to find the min value in c array (Prefix Sum logic)

int main(){
	pthread_t thread[num_of_threads]; 	//thread pool
	pthread_mutex_init(&pmutex,NULL);	//initializing lock

	//Initializing our a and b arrays
	for(int j = 0; j<n;j++){	
		a[j] = rand() %100;
		b[j] = rand() %100;
	}

	long i;
	for(i =	0; i< num_of_threads; i++){	//Fork
		pthread_create(&thread[i], NULL, mySum, (void*) i );
		sleep(1);	//pause for 1 sec
	}
	for(i =	0; i< num_of_threads; i++)	//Join
		pthread_join(thread[i], NULL);

	for(i =	0; i< num_of_threads; i++)	//Fork
		pthread_create(&thread[i], NULL, minVal, (void*) i );
	
	for(i =	0; i< num_of_threads; i++)	//Join
		pthread_join(thread[i], NULL);

	printf("the min value is: %d\n", c[n-1]); //print the min value in the array
	return 0;
}

void *mySum(void* threadid){		//function to add our arrays
	printf("Hello from thread: %ld\n", (long) threadid);
	long idx = (long)threadid; 	//cast threadid variable
	while(idx < n){
		c[idx] = a[idx] + b[idx]; 	//store sum
		printf("%d, ", c[idx]);		
		idx+=num_of_threads;		//increment thread index by adding total number of threads
	}
	printf("\n");
	return NULL;
}

void *minVal(void* threadid){		//function to find the min value in c array
	int size = n/num_of_threads;	//number of items that each thread will compute
	int idx;	//index
	long thread_id = (long)threadid; //cast threadid variable
	for(long i = 0; i<size; i++){
		idx = thread_id*size+i;	//increment index
		if(i == size-1){	// last iteration 
			pthread_mutex_lock(&pmutex);	//Thread locks the value to compare (One thread at a time)
			if (c[idx] < c[n-1])
				c[n-1] = c[idx];	//c[n-1] will hold the min value in the array
			pthread_mutex_unlock(&pmutex);	//Thread releases the lock
		}
		else if(c[idx] < c[thread_id*size+(size-1)]){ 	//c[thread_id*size+(size-1)] holds each thread's min value
			c[thread_id*size+(size-1)] = c[idx];
		}
	}
	return NULL;
}
