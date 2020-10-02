#include <iostream>
#include<thread>
using namespace std;

void myPrint(int myrank);
float f(float x);

int uni_num_threads;
float myresult;

int main()
{
	int num_threads = 6;
	uni_num_threads = num_threads;

	thread* t = new thread[num_threads];

	for(int i = 0; i < num_threads; i++)
	{
		t[i] = thread(myPrint, i);
		this_thread::sleep_for(5ms);
	}

	for(int i = 0; i < num_threads; i++)
	{
		t[i].join();
	}

	cout << "pi = " << myresult << endl;
	return 0;
}
void myPrint(int myrank)
{
	float a = 0.0;
	float b =1.0;
	float n = 1000000;
	float h = (b-a)/n;
	float local_n = n/uni_num_threads;
	float local_a = a + myrank*local_n*h;
	float local_b = local_a + local_n*h;
	//myresult = (f(local_a) + f(local_b)) / 2.0;
	float x;

	for(int i = 1; i <= local_n; i++)
	{
		x = local_a + i*h;
		myresult += f(x);
	}
	myresult += myresult*h;

}

float f(float x)
{
	float sol;
	sol = (4.0) / ((1+(x*x)));
	return sol;
}