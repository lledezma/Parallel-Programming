#include "stdafx.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdio.h>
#include <stdlib.h>
// #include <windows.h>

void PI(double a, double b, int numIntervals, double* global_result_p);
int main(int argc, char* argv[])
{
	double   global_result = 0.0;
	volatile DWORD dwStart;
	int numThreads = strtol(argv[1], NULL, 10);
	dwStart = GetTickCount();

#  pragma omp parallel num_threads(1)
	PI(0, 1, 10000000, &global_result);
	printf("number of threads %d \n", 1);
	printf("numberInterval %d \n", 10000000);
	printf("Pi = %f \n", global_result);
	printf_s("milliseconds %d \n", GetTickCount() - dwStart);
	printf("\n");
	dwStart = GetTickCount();
	global_result = 0.0;

#  pragma omp parallel num_threads(1)
	PI(0, 1, 100000000, &global_result);
	printf("number of threads %d \n", 1);
	printf("numberInterval %d \n", 100000000);
	printf("Pi = %f \n", global_result);
	printf_s("milliseconds %d \n", GetTickCount() - dwStart);
	printf("\n");
	dwStart = GetTickCount();
	global_result = 0.0;

#  pragma omp parallel num_threads(1)
	PI(0, 1, 1000000000, &global_result);
	printf("number of threads %d \n", 1);
	printf("numberInterval %d \n", 1000000000);
	printf("Pi = %f \n", global_result);
	printf_s("milliseconds %d \n", GetTickCount() - dwStart);
	printf("\n");
	dwStart = GetTickCount();
	global_result = 0.0;

#  pragma omp parallel num_threads(2)
	PI(0, 1, 10000000, &global_result);
	printf("number of threads %d \n", 2);
	printf("numberInterval %d \n", 10000000);
	printf("Pi = %f \n", global_result);
	printf_s("milliseconds %d \n", GetTickCount() - dwStart);
	printf("\n");
	dwStart = GetTickCount();
	global_result = 0.0;

#  pragma omp parallel num_threads(2)
	PI(0, 1, 100000000, &global_result);
	printf("number of threads %d \n", 2);
	printf("numberInterval %d \n", 100000000);
	printf("Pi = %f \n", global_result);
	printf_s("milliseconds %d \n", GetTickCount() - dwStart);
	printf("\n");
	dwStart = GetTickCount();
	global_result = 0.0;

#  pragma omp parallel num_threads(2)
	PI(0, 1, 1000000000, &global_result);
	printf("number of threads %d \n", 2);
	printf("numberInterval %d \n", 1000000000);
	printf("Pi = %f \n", global_result);
	printf_s("milliseconds %d \n", GetTickCount() - dwStart);
	printf("\n");
	dwStart = GetTickCount();
	global_result = 0.0;

#  pragma omp parallel num_threads(4)
	PI(0, 1, 10000000, &global_result);
	printf("number of threads %d \n", 4);
	printf("numberInterval %d \n", 10000000);
	printf("Pi = %f \n", global_result);
	printf_s("milliseconds %d \n", GetTickCount() - dwStart);
	printf("\n");
	dwStart = GetTickCount();
	global_result = 0.0;

#  pragma omp parallel num_threads(4)
	PI(0, 1, 100000000, &global_result);
	printf("number of threads %d \n", 4);
	printf("numberInterval %d \n", 100000000);
	printf("Pi = %f \n", global_result);
	printf_s("milliseconds %d \n", GetTickCount() - dwStart);
	printf("\n");
	dwStart = GetTickCount();
	global_result = 0.0;

#  pragma omp parallel num_threads(4)
	PI(0, 1, 1000000000, &global_result);
	printf("number of threads %d \n", 4);
	printf("numberInterval %d \n", 1000000000);
	printf("Pi = %f \n", global_result);
	printf_s("milliseconds %d \n", GetTickCount() - dwStart);
	printf("\n");
	dwStart = GetTickCount();
	global_result = 0.0;

#  pragma omp parallel num_threads(numThreads)
	PI(0, 1, 10000000, &global_result);
	printf("number of threads %d \n", numThreads);
	printf("numberInterval %d \n", 10000000);
	printf("Pi = %f \n", global_result);
	printf_s("milliseconds %d \n", GetTickCount() - dwStart);
	printf("\n");
	dwStart = GetTickCount();
	global_result = 0.0;

#  pragma omp parallel num_threads(numThreads)
	PI(0, 1, 100000000, &global_result);
	printf("number of threads %d \n", numThreads);
	printf("numberInterval %d \n", 100000000);
	printf("Pi = %f \n", global_result);
	printf_s("milliseconds %d \n", GetTickCount() - dwStart);
	printf("\n");
	dwStart = GetTickCount();
	global_result = 0.0;

#  pragma omp parallel num_threads(numThreads)
	PI(0, 1, 1000000000, &global_result);
	printf("number of threads %d \n", numThreads);
	printf("numberInterval %d \n", 1000000000);
	printf("Pi = %f \n", global_result);
	printf_s("milliseconds %d \n", GetTickCount() - dwStart);
}
void PI(double a, double b, int numIntervals, double* global_result_p)
{

	int i;
	double x, my_result, sum = 0.0, interval, local_a, local_b, local_numIntervals;
	int myThread = omp_get_thread_num();
	int numThreads = omp_get_num_threads();
	interval = (b - a) / (double)numIntervals;
	local_numIntervals = numIntervals / numThreads;
	local_a = a + myThread * local_numIntervals * interval;
	local_b = local_a + local_numIntervals * interval;
	sum = 0.0;
	for (i = 0; i < local_numIntervals; i++)
	{
		x = local_a + i * interval;
		sum = sum + 4.0 / (1.0 + x * x);
	};
	my_result = interval * sum;
#  pragma omp critical
	* global_result_p += my_result;
}
