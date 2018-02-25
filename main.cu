#pragma once
#include <stdio.h>
#include <iostream>

#include "processing.h"

#include <vector>
#include <memory>
#include "KernelSlowConvolution.cu"
#include "Filter.cu"
#include "CpuSlowConvolution.cpp"
#include <vector>
#include <algorithm>

using namespace cv;
using namespace std;
using namespace processing;

static const string INPUT_IMAGE_PATH = "input_img.jpg";
static const string OUTPUT_IMAGE_PATH = "output_img.jpg";



int main()
{
	allocateMemmoryDevice<uchar>(1);
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
		(3,
		{
			0,0.2,0,
			0.2,0.2,0.2,
			0, 0.2, 0
		}
			
	);
	auto sobel = createFilter<COUNT_TYPE>
	(3, 
	{
		-1.0,0.0,1,
		-2.0,0.0,2.0,
		-1.0,0.0,1.0
	}, 10.0
	);
	vector< shared_ptr<AbstractFilter<COUNT_TYPE> > > filters = {  sobel, gausianBlur, gausianBig };
	for (size_t i = 0; i < gausianBlur->getWidth() * gausianBlur->getWidth(); i++)
	{
		cout << ((Filter<COUNT_TYPE, 3> *) gausianBlur.get())->getFilter()[i] << endl;
	}
	KernelSlowConvolution<COUNT_TYPE> kernel(filters);
	CpuSlowConvolution<COUNT_TYPE> cpuKonv(filters);
	ImageFactory factory(INPUT_IMAGE_PATH);
	
	vector<shared_ptr<COUNT_TYPE>> resultsKernel;
	auto timeGPU = factory.run(&kernel, resultsKernel);
	vector<shared_ptr<COUNT_TYPE>> resultsCpu;
	auto timeCPU = factory.run(&cpuKonv, resultsCpu);
	cout << "konec" ;
	cout << "Time GPU: " << timeGPU.getTimeMicro() << endl;
	cout << "Time CPU: " << timeCPU.getTimeMicro() << endl;
	for (uint i = 0; i < std::min(resultsKernel.size(), resultsCpu.size()) ; i++)
	{
		auto pGPU = resultsKernel[i];
		auto pCPU = resultsCpu[i];
		for (size_t i = 0; i < factory.getNumPixels(); i++) 
		{
			if ( [&pGPU, &pCPU, i] () {
				return fabs(pGPU.get()[i] - pCPU.get()[i]) > 0.001;
			}() )
			{
				cout << "-----------------------" << endl;
				cout << "Index: " << i << endl;
				cout << "GPU " << pGPU.get()[i] << endl;
				cout << "CPU " << pCPU.get()[i] << endl;
				cout << "-----------------------" << endl;
			}
		}
	}
	//factory.saveRGBAImg(OUTPUT_IMAGE_PATH);

	//imwrite(OUTPUT_IMAGE_PATH, factory.getHostGray());
	return 0;
}