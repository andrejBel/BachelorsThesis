#include <stdio.h>
#include <iostream>

#include "processing.h"

#include <vector>
#include <memory>
#include "KernelSlowConvolution.cu"
#include "Filter.cu"
#include "CpuSlowConvolution.cpp"



using namespace cv;
using namespace std;
using namespace processing;

static const string INPUT_IMAGE_PATH = "input_img.jpg";
static const string OUTPUT_IMAGE_PATH = "output_img.jpg";



int main()
{
	#define COUNT_TYPE double
	auto gausianBig = createFilter<COUNT_TYPE>(
		7, {
			0.00000067,	0.00002292,	0.00019117,	0.00038771,	0.00019117,	0.00002292,	0.00000067,
			0.00002292,	0.00078634,	0.00655965,	0.01330373,	0.00655965,	0.00078633,	0.00002292,
			0.00019117,	0.00655965,	0.05472157,	0.11098164,	0.05472157,	0.00655965,	0.00019117,
			0.00038771,	0.01330373,	0.11098164,	0.22508352,	0.11098164,	0.01330373,	0.00038771,
			0.00019117,	0.00655965,	0.05472157,	0.11098164,	0.05472157,	0.00655965,	0.00019117,
			0.00002292,	0.00078633,	0.00655965,	0.01330373,	0.00655965,	0.00078633, 0.00002292,
			0.00000067,	0.00002292,	0.00019117,	0.00038771,	0.00019117,	0.00002292,	0.00000067
		},
		1.0
		);
	auto gausianBlur = createFilter<COUNT_TYPE>
		(5,
		{
			1.0,4,6,4,1,
			4,16,24,16,4,
			6,24,36,24,6,
			4,16,24,16,4,
			1,4,6,4,1
		},
			1.0 / 256.0
	);
	auto sobel = createFilter<COUNT_TYPE>
	(3, 
	{
		-1.0,0.0,1,
		-2.0,0.0,2.0,
		-1.0,0.0,1.0
	}, 0.5
	);
	vector< shared_ptr<AbstractFilter<COUNT_TYPE> > > filters = { gausianBlur };
	for (size_t i = 0; i < 5*5; i++)
	{
		cout << ((Filter<COUNT_TYPE, 5> *) gausianBlur.get())->getFilter()[i] << endl;
	}
	KernelSlowConvolution<COUNT_TYPE> kernel(filters);
	CpuSlowConvolution<COUNT_TYPE> cpuKonv(filters);
	ImageFactory factory(INPUT_IMAGE_PATH);
	

	factory.run(&cpuKonv);
	
	//factory.copyDeviceGrayToHostGray();

	imwrite(OUTPUT_IMAGE_PATH, factory.getHostGray());
	return 0;
}