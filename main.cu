#pragma once
#include <stdio.h>
#include <iostream>

#include "processing.h"

#include <vector>
#include <memory>

#include "KernelSlowConvolution.cu"
#include "KernelSlowConvolutionNoEdgeCopy.cu"
#include "KernelSharedMemory.cu"
#include "KernelSharedMemoryManaged.cu"
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

void sleep() 
{
	this_thread::sleep_for(chrono::milliseconds(2000));
}

__global__ void kernel(float *x, int n)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
		x[i] = sqrt(pow(3.14159, i));
	}
}

const int N =( 1 << 20) * 10;

void launch_kernel()
{
	float *data;
	cudaMalloc(&data, N * sizeof(float));

	kernel << <1, 64 >> >(data, N);

	//cudaStreamSynchronize(0);
}


int main()
{
	allocateMemmoryDevice<uchar>(1);


	//Test<float>::testAllAgainstCpu();
	//Test<float>::testAgainstCpuSingleCore(make_shared<KernelSharedMemory<float>>(), 1);
	//cudaDeviceSetCacheConfig(cudaFuncCache::cudaFuncCachePreferL1);
	//Test<float>::testAloneForManaged(make_shared<CpuSlowConvolution<float>>(), 2);
	Test<float>::Builder()
	.addFilter(Test<float>::get3x3Filter())
	.addFilter(Test<float>::get3x3Filter())
	.addFilter(Test<float>::get3x3Filter())
	.addFilter(Test<float>::get3x3Filter())
	.addFilter(Test<float>::get3x3Filter())
	.addFilter(Test<float>::get3x3Filter())
	.setReplications(10)
	.addRunnable(make_shared<KernelSharedMemory<float>>()).
	build()();
	
	//Test<float>::testAlone(make_shared<CpuSlowConvolution<float>>(), 2);
	//Test<float>::testAlone(make_shared<CpuSlowConvolution<float>>(), 2);
	
	//Test<float>::testAlone(make_shared<KernelSharedMemory<float>>(), 1);
	//Test<float>::testAloneForManaged(make_shared<KernelSharedMemory<float>>(), 10);
	//Test<float>::testAloneForManaged(make_shared<KernelSlowConvolutionNoEdgeCopy<float>>(), 10);
	//Test<float>::testAloneForManaged(make_shared<KernelSharedMemory<float>>(), 5);
	//cudaDeviceReset();
	return 0;
}

