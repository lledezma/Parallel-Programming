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

int BruteForceMatch(char* str1, char* str2);


int main(int argc, char* argv[])
{
	char* sequence = T(1000000);
	char* pattern = P(8);
	volatile DWORD dwStart;
	double  global_result = 0;
	int numThreads = strtol(argv[1], NULL, 10);

	printf("bruteforcematch that returns number of matches: \n");
	dwStart = GetTickCount64();
	global_result = BruteForceMatch(sequence, pattern);
	printf("total matches: %f \n", global_result);
	printf_s("milliseconds: %d\n", GetTickCount64() - dwStart);
}

//BRUTEFORCEMATCH METHOD
int BruteForceMatch(char* str1, char* str2) {
	int count = 0;
	int m = strlen(str2);
	int n = strlen(str1);

	if (n >= m) {
		for (int i = 0; i <= n - m; i++) {
			int j = 0;
			while ((j < m) && (str1[i + j] == str2[j])) {
				j = j + 1;
			}
			if (j == 8)	{
				count = count + 1;
			}
		}
	}
	return count;
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
