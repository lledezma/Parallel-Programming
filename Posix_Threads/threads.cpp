			//Adding two arrays and finding the min value. Using prefix sum algorithm logic
			//to find min value. Multithreading program using POSIX Threads
#include <iostream>
#include <thread>
#include <mutex>
using namespace std;


int a[100];
int b[100];
int c[100];

mutex mylock;

int num_threads = 5;
int n = sizeof(a)/sizeof(a[0]);  //size of array
int minval = 1000;

void mySum(int thread_id, int size);
void myMin(int thread_id, int size);

int main()
{
	
	thread* t = new thread[num_threads];

	int size = n/num_threads;		//size of partition


	for(int i = 0; i < 100; i++){
		a[i] = rand() %100;
		b[i] = rand() %100;
	}

	for(int i = 0; i < num_threads; i++){
		t[i] = thread(mySum, i, size);
		this_thread::sleep_for(3ms);
	}

	for(int i = 0; i < num_threads; i++){
		t[i].join();
	}


	for(int i = 0; i < num_threads; i++){
		t[i] = thread(myMin, i, size);
	}
	for(int i = 0; i < num_threads; i++){
		t[i].join();
	}

	minval = c[n-1];
	cout << "the min value = " << minval << endl;
	return 0;

}

void mySum(int thread_id, int size){	//function to add our arrays
	int idx;
	mylock.lock();
	cout << "Hello from thread: " << thread_id << endl;
	for(int i =0; i < size;i++)
	{
		idx = thread_id*size+i;
		c[idx] = a[idx] + b[idx];
		cout << a[idx] << " + " << b[idx] << " = " << c[idx] << " | " << "........" << idx << endl;
		this_thread::sleep_for(5ms);
	}
	mylock.unlock();
}

void myMin(int thread_id, int size) //function to find the min value in c array
{
	int idx;
	mylock.lock();
	for(int i =0; i < size;i++)
	{
		idx = thread_id*size+i;
		if(i == size-1)
		{
			if(c[idx] < c[n-1]){
				c[n-1] = c[idx];
			}
		}
		else if (c[idx] < c[thread_id*size+size-1])
		{
			c[thread_id*size+size-1] = c[idx];
		}
		this_thread::sleep_for(5ms);
	}
	mylock.unlock();
}