                ////Using Numerical Integration to Calculate PI = π
#include <iostream>
#include <omp.h> 
#include <StopWatch.h> 
#include <tgmath.h> 

using namespace std;

int main()
{
    StopWatch sw;
    int Nums[] = {10,100,500,100000,1000000,100000000};
    for (int h = 1; h <= 4; h++) {
        printf("Using number of threads: %d\n", h);
        for (int j = 0; j < 6; j++) {
            int a = 0;
            int b = 1;
            int N = Nums[j];
            double stepSize = (double)(b - a) / N;
            double intergrationHeights = 0;
            double x;
            sw.start();
         #pragma omp parallel for num_threads(h) reduction(+:intergrationHeights)
            for (int i = 0; i < N; i++) {
                x = i * stepSize + a;
                intergrationHeights += 4 / (1 + (x * x));
            }
            double results = intergrationHeights * stepSize;
            sw.stop();
            printf("Estimated PI=%1f. Estimated Time = %1f ms\n", results, sw.elapsedTime());
        }
    }
    return 0;
}
