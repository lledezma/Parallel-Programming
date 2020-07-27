		//Add two arrays and store the new values in a new array. Calculate the min value in
		// the new generated array.
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

void mySum(int thread_id, int size){
	mylock.lock();
	cout << "Hello from thread: " << thread_id << endl;
	
	for(int i =0; i < size;i++)
	{
		c[thread_id*size+i] = a[thread_id*size+i] + b[thread_id*size+i];
		cout << a[thread_id*size+i] << " + " << b[thread_id*size+i] << " = " << c[thread_id*size+i] << " | " << "........" << thread_id*size+i << endl;
		this_thread::sleep_for(5ms);
	}
	mylock.unlock();
}

void myMin(int thread_id, int size)
{
	mylock.lock();
	for(int i =0; i < size;i++)
	{
		if(i == size-1)
		{
			if(c[thread_id*size+i] < c[n-1]){
				c[n-1] = c[thread_id*size+i];
			}
		}
		else if (c[thread_id*size+i] < c[thread_id*size+size-1])
		{
			c[thread_id*size+size-1] = c[thread_id*size+i];
		}
		this_thread::sleep_for(5ms);
	}
	mylock.unlock();
}

// for(int i = 0; i < size; i++)
// {
// 	reveseArray[i] = myArray[n-1-(thread_id*size+i)];
// }