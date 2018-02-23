#include <stdio.h>
#include <iostream>

#include "processing.h"

#include <vector>
#include "KernelSlowConvolution.cu"
#include "Filter.cu"




using namespace cv;
using namespace std;
using namespace processing;

static const string INPUT_IMAGE_PATH = "input_img.jpg";
static const string OUTPUT_IMAGE_PATH = "output_img.jpg";



int main()
{
	vector< Filter<float> > filters;
	filters.push_back(Filter<float>(3, 3, {
											1.0,1.0,1.0,
											1.0,1.0,1.0,
											1.0,1.0,1.0
										  }, 1.0 / 9.0));
	KernelSlowConvolution<float> kernel(filters);
	ImageFactory factory(INPUT_IMAGE_PATH);
	

	
	factory.run(&kernel);
	
	//factory.copyDeviceGrayToHostGray();

	imwrite(OUTPUT_IMAGE_PATH, factory.getHostGray());
	return 0;
}