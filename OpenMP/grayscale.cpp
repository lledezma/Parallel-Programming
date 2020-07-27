				//Converting scene.bmp, into a grayscale picture using OPENMP
#include <iostream>
#include "BmpImage.h"
#include "StopWatch.h"
#include<omp.h>

int main() {
	StopWatch sw;
	BmpImage image;
	image.load("scene.bmp");
	printf("----------------------------------------------\n");
	int pixelCount = image.getTotalPixelCount();
	printf("Image Total Pixel Count = %d\n", image.getTotalPixelCount());


	for (int h = 2; h <= 4; h++) {
		int size = pixelCount / h;
		/*
			RGB frstPixel = image.getPixel(0);
			printf("First pixel value (Bottom Left) (R:%d,G:%d,B:%d)\n", frstPixel.R, frstPixel.G, frstPixel.B);

			RGB lastPixel = image.getPixel(pixelCount - 1);
			printf("Last pixel value (Bottom Left) (R:%d,G:%d,B:%d)\n", lastPixel.R, lastPixel.G, lastPixel.B);

			printf("----------------------------------------------\n");
			printf("Set First Pixel Value Full White\n");
			image.setPixel(0, 255);
			frstPixel = image.getPixel(0);
			printf("First pixel value (Bottom Left) (R:%d,G:%d,B:%d)\n", frstPixel.R, frstPixel.G, frstPixel.B);
			printf("----------------------------------------------\n");
		*/
		printf("Total Number of threads: %d\n", h);
		
		sw.start();
	#  pragma omp parallel num_threads(h) 
		for (int i = 0; i < size; i++)
		{
			double frstPixel = image.getPixel(omp_get_thread_num()*size+i).R * 0.3 + image.getPixel(omp_get_thread_num() * size + i).G * 0.59 + image.getPixel(omp_get_thread_num() * size + i).B * 0.11;
			image.setPixel(omp_get_thread_num() * size + i, frstPixel);
		}
		sw.stop();
		printf("Save File Estimated time:  %1f ms\n", sw.elapsedTime());
		image.save("test_new.bmp");
	}

	return 0;
}