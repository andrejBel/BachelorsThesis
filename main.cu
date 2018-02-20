#include <stdio.h>
#include <iostream>

#include "processing.h"

#include "GpuTimer.h"
#include "KernelBlackImage.h"




using namespace cv;
using namespace std;
using namespace processing;

static const string INPUT_IMAGE_PATH = "input_img.jpg";
static const string OUTPUT_IMAGE_PATH = "output_img.jpg";



int main()
{
	ImageFactory factory(INPUT_IMAGE_PATH);
	KernelBlackImage kernel;
	factory.run(&kernel);
	
	factory.copyDeviceGrayToHostGray();
	//checkCudaErrors(cudaDeviceSynchronize());
	imwrite(OUTPUT_IMAGE_PATH, factory.getHostGray());
	return 0;
}