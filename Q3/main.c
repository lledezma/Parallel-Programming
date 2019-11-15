#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <windows.h>

#define number 1000
int A[number][number];
int B[number][number];
int C[number][number];

void generatematrices();
void multiplication(int(*A)[number], int(*B)[number], int(*C)[number], int n);

int main(int argc, char* argv[])
{
	generatematrices();

	volatile DWORD dwStart;
	double  global_result = 0;
	int numThreads = strtol(argv[1], NULL, 10);
	dwStart = GetTickCount64();



#  pragma omp parallel num_threads(1)
	multiplication(A, B, C, number);
	printf("threads %d \n", 1);
	printf_s("milliseconds: %d\n", GetTickCount64() - dwStart);
	dwStart = GetTickCount64();


#  pragma omp parallel num_threads(2)
	multiplication(A, B, C, number);
	printf("threads %d \n", 2);
	printf_s("milliseconds: %d\n", GetTickCount64() - dwStart);
	dwStart = GetTickCount64();


#  pragma omp parallel num_threads(4)
	multiplication(A, B, C, number);
	printf("threads %d \n", 4);
	printf_s("milliseconds: %d\n", GetTickCount64() - dwStart);
	dwStart = GetTickCount64();


#  pragma omp parallel num_threads(numThreads)
	multiplication(A, B, C, number);
	printf("threads %d \n", 8);
	printf_s("milliseconds: %d\n", GetTickCount64() - dwStart);

	printf("A-MATRIX \n");
	for (int i = 0; i < number; i++) {
		for (int j = 0; j < number; j++) {
			printf("%d  ", A[i][j]);
		}
		printf("\n");
	}

	printf("B-MATRIX \n");
	for (int i = 0; i < number; i++) {
		for (int j = 0; j < number; j++) {
			printf("%d  ", B[i][j]);
		}
		printf("\n");
	}

	printf("C-MATRIX \n");
	for (int i = 0; i < number; i++) {
		for (int j = 0; j < number; j++) {
			printf("%d  ", C[i][j]);
		}
		printf("\n");
	}
}


void multiplication(int(*A)[number], int(*B)[number], int(*C)[number], int n)
{

	int mythread = omp_get_thread_num();
	int numthreads = omp_get_num_threads();
	int totalsum = 0;
	for (int i = mythread; i < n; i += numthreads)
	{
		for (int h = 0; h < n; h++)
		{

			totalsum += (A[i][h] * B[h][mythread]);
		}
		C[i][mythread] = totalsum;
		totalsum = 0;
		for (int j = mythread + 1; j % n != mythread; j++)
		{

			for (int x = 0; x < n; x++)
			{

				totalsum += (A[i][x] * B[x][j]);
			}
			C[i][j] = totalsum;
			totalsum = 0;
		}
	}

}

void generatematrices()
{
	for (int i = 0; i < number; i++)
	{
		for (int j = 0; j < number; j++)
		{

			A[i][j] = (rand() % 20) + 1;
			B[i][j] = (rand() % 20) + 1;
		}
	}
}
