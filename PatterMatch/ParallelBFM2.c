//#include "stdafx.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <string.h>
#include <time.h>

char* T(int length);
char* P(int length);

void parallelBruteforcematch(char* str1, char* str2, int q, int m, double* global_result_p);

int main(int argc, char* argv[])
{
	char* sequence = T(1000000);
	char* pattern = P(8);

	volatile DWORD dwStart;
	double  global_result = 0;
	int numThreads = strtol(argv[1], NULL, 10);
	
	printf("PARALLELBRUTEFORCEMATCH: \n");

	global_result = 0;
	dwStart = GetTickCount64();


#  pragma omp parallel num_threads(1)
	parallelBruteforcematch(sequence, pattern, 1000000, 8, &global_result);
	printf("number of threads %d \n", 1);
	printf("total matches: %f \n", global_result);
	printf_s("milliseconds: %d\n", GetTickCount64() - dwStart);
	global_result = 0;
	dwStart = GetTickCount64();

#  pragma omp parallel num_threads(2)
	parallelBruteforcematch(sequence, pattern, 1000000, 8, &global_result);
	printf("number of threads %d \n", 2);
	printf("total matches: %f \n", global_result);
	printf_s("milliseconds: %d\n", GetTickCount64() - dwStart);
	global_result = 0;
	dwStart = GetTickCount64();

#  pragma omp parallel num_threads(4)
	parallelBruteforcematch(sequence, pattern, 1000000, 8, &global_result);
	printf("number of threads %d \n", 4);
	printf("total matches: %f \n", global_result);
	printf_s("milliseconds: %d\n", GetTickCount64() - dwStart);
	global_result = 0;
	dwStart = GetTickCount64();

#  pragma omp parallel num_threads(numThreads)
	parallelBruteforcematch(sequence, pattern, 1000000, 8, &global_result);
	printf("number of threads %d \n", numThreads);
	printf("total matches: %f \n", global_result);
	printf_s("milliseconds: %d\n", GetTickCount64() - dwStart);

}

//PARALLEL BRUTE FORCE MATCH
void parallelBruteforcematch(char* str1, char* str2, int q, int m, double* global_result_p)
{

	double my_result = 0;
	int myThread = omp_get_thread_num();
	int numThreads = omp_get_num_threads();

	for (int i = myThread - 1; i < q; i += numThreads)
	{
		if (q - i + 1 >= m)
		{
			for (int j = 0; j < m && str1[i + j] == str2[j]; j++)
			{
				if (j == m - 1)
				{
					my_result += 1;
				}
			}
		}
		else
		{
			i = q;
		}
	}
#  pragma omp critical
	* global_result_p += my_result;
}

//SEQUENCE generator
char* T(int length) {
	char* string = "acgt";
	size_t stringLen = 4;
	char* randomString;

	randomString = malloc(sizeof(char) * (length + 1));

	if (!randomString) {
		return (char*)0;
	}

	unsigned int key = 0;

	for (int n = 0; n < length; n++) {
		key = rand() % stringLen;
		randomString[n] = string[key];
	}

	randomString[length] = '\0';

	return randomString;
}

//PATTERN generator
char* P(int length) {
	char* string = "acgt";
	size_t stringLen = 4;
	char* randomString;

	randomString = malloc(sizeof(char) * (length + 1));

	if (!randomString) {
		return (char*)0;
	}

	unsigned int key = 0;

	for (int n = 0; n < length; n++) {
		key = rand() % stringLen;
		randomString[n] = string[key];
	}

	randomString[length] = '\0';

	return randomString;
}
