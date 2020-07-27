#include <iostream>
#include <omp.h> 
#include <StopWatch.h> 
#include <tgmath.h> 
using namespace std;
            //Monte Carlo Method using OpenMP
int main()
{
    StopWatch sw;
   for (int h = 2; h <= 5; h++)
   {
            double N = 100000000;        // number of tosses
            double number_in_circle = 0.0;
            float a,b;
            float distance_squared;
            printf("Using number of threads: %d\n", h);
            sw.start();
        #pragma omp parallel for num_threads(h) reduction(+:number_in_circle)
            for (int i = 0; i < N; i++) {
                a = (1 - 0) * (rand() / (double)RAND_MAX) + 0;
                b = (1 - 0) * (rand() / (double)RAND_MAX) + 0;
                distance_squared = a * a + b * b;
                if (distance_squared <= 1.0)
                    number_in_circle = number_in_circle + 1;
            }

            float results = 4 * number_in_circle / N;

            sw.stop();
           printf("Estimated PI=%1f. Estimated Time = %1f ms\n", results, sw.elapsedTime());
    
   }
    return 0;
}