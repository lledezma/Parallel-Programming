#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

//Global variables and Shared variables
pthread_mutex_t pmutex;

int  a[100];
int  b[100];
int  c[100];
int num_of_threads = 4;
int n = sizeof(a)/sizeof(a[0]);

void *mySum(void* threadid) ;
void *minVal(void* threadid);

int main(){
	pthread_t thread[num_of_threads];
	pthread_mutex_init(&pmutex,NULL);

	for(int j = 0; j<n;j++){
		a[j] = rand() %100;
		b[j] = rand() %100;
	}

	long i;
	for(i =	0; i< num_of_threads; i++){
		pthread_create(&thread[i], NULL, mySum, (void*) i );
		sleep(1);
	}

	for(i =	0; i< num_of_threads; i++)
		pthread_join(thread[i], NULL);

	for(i =	0; i< num_of_threads; i++)
		pthread_create(&thread[i], NULL, minVal, (void*) i );
	
	for(i =	0; i< num_of_threads; i++)
		pthread_join(thread[i], NULL);

	printf("the min value is: %ld\n", c[n-1]);
	return 0;
}

void *mySum(void* threadid){
	printf("Hello from thread: %ld\n", (long) threadid);
	int size = n/num_of_threads;
	for(long i = 0; i < size; i++){
		c[(long)threadid * size + i] = a[(long)threadid * size + i] + b[(long)threadid * size + i];
		printf("%ld , ", c[(long)threadid * size + i] );
	}
	printf("\n");
	return NULL;
}

void *minVal(void* threadid){
	int size = n/num_of_threads;
	for(long i = 0; i<size; i++){
		if(i == size-1){
			pthread_mutex_lock(&pmutex);
			if (c[(long)threadid*size+i] < c[n-1])
				c[n-1] = c[(long)threadid*size+i];
			pthread_mutex_unlock(&pmutex);
		}
		else if(c[(long)threadid*size+i] < c[(long)threadid*size+(size-1)]){
			c[(long)threadid*size+(size-1)] = c[(long)threadid*size+i];
		}
	}
	return NULL;
}
