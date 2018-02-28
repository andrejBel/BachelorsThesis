#pragma once
#include <stdio.h>
#include <iostream>

#include "processing.h"

#include <vector>
#include <memory>

#include "KernelSlowConvolution.cu"
#include "KernelSlowConvolutionNoEdgeCopy.cu"
#include "KernelSharedMemory.cu"
#include "KernelSharedMemoryAsync.cu"
#include "Filter.cu"
#include "CpuSlowConvolution.cpp"
#include "CPUSlowConvolutionAsync.cpp"
#include <vector>
#include <algorithm>
#include "Test.h"
#include "ThreadPool.h"
using namespace cv;
using namespace std;
using namespace processing;


static const string OUTPUT_IMAGE_PATH = "output_img.jpg";




int main()
{
	allocateMemmoryDevice<uchar>(1);
	//Test<float>::testAllAgainstCpu();
	Test<float>::testAloneForManaged(make_shared<KernelSharedMemoryAsync<float>>(), 10);
	Test<float>::testAloneForManaged(make_shared<KernelSharedMemory<float>>(), 10);
	Test<float>::testAloneForManaged(make_shared<KernelSlowConvolutionNoEdgeCopy<float>>(), 10);
	//Test<float>::testAloneForManaged(make_shared<KernelSharedMemory<float>>(), 5);
	//cudaDeviceReset();
	return 0;
}

