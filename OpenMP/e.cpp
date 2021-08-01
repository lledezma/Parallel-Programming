#include <iostream>
#include <omp.h>
#include <StopWatch.h>
#include <math.h>
using namespace std;
                            //Computing the Mathematical Constant e=2.71828 using OpenMP
signed  long long factorial(int n);
int main()
{
    int samples[] = {100,500,1000,10000,30000};
    for (int h = 1; h<= 6; h++) {
        for(int z = 0; z < 5; z++) {
            StopWatch sw;
            float localvalue =0;
            double x = 1.0;
            int size = samples[z];
            sw.start();
            printf("Total number of threads: %d\n", h);
            printf("Total of partions: %d\n", samples[z]);
        # pragma omp parallel for num_threads(h) reduction(+:localvalue)
            for(int n = 0; n < size; n++){
                localvalue += pow(x,n) / (factorial(n));
            }
            sw.stop();
            printf("Estimated Euler’s number=%1f. Estimated Time = %1f ms\n", localvalue, sw.elapsedTime());
            printf("\n");
        }
    }
    return 0;
}

signed long long factorial(int n) {
    long double result = 1;
    if(n == 0)
        return result;
    for(int i = 1; i <=n; ++i) {
        result *= i;
    }
    return result;
}
