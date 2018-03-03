#pragma once
#include <stdio.h>
#include <iostream>

#include "processing.h"

#include <vector>
#include <memory>

#include "KernelNaive.cu"
#include "KernelSlowConvolution.cu"
#include "KernelSlowConvolutionNoEdgeCopy.cu"
#include "KernelSharedMemory.cu"
#include "KernelSharedMemoryManaged.cu"
#include "KernelSharedMemoryAsync.cu"

#include "Filter.cu"
#include "CpuSlowConvolution.cpp"
#include "CPUSlowConvolutionAsync.cpp"

#include <vector>
#include <algorithm>
#include "Test.h"
#include "ThreadPool.h"

#include <cuda.h>
#include <helper_cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "opencv2/core/utility.hpp"

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

__global__ void kernel(float * input, float value, int n) 
{
	for (size_t i = 0; i < n  ; i++)
	{
		input[i] = value;
	}
}


int main()
{

	allocateMemmoryDevice<uchar>(1);
	

	
	//Test<float>::testAllAgainstCpu();
	//Test<float>::testAgainstCpuSingleCore(make_shared<KernelSharedMemory<float>>(), 1);
	//cudaDeviceSetCacheConfig(cudaFuncCache::cudaFuncCachePreferL1);
	//Test<float>::testAloneForManaged(make_shared<CpuSlowConvolution<float>>(), 2);


#define TYPE float
	Test<TYPE>::testAgainstCpuMulticore(make_shared<KernelSharedMemory<TYPE>>());
	Test<TYPE>::testAgainstCpuMulticore(make_shared<KernelSharedMemoryManaged<TYPE>>());
	//Test<float>::testAlone(make_shared<CpuSlowConvolution<float>>(), 2);
	//Test<float>::testAlone(make_shared<CpuSlowConvolution<float>>(), 2);
	
	//Test<float>::testAlone(make_shared<KernelSharedMemory<float>>(), 1);
	//Test<float>::testAloneForManaged(make_shared<KernelSharedMemory<float>>(), 10);
	//Test<float>::testAloneForManaged(make_shared<KernelSlowConvolutionNoEdgeCopy<float>>(), 10);
	//Test<float>::testAloneForManaged(make_shared<KernelSharedMemory<float>>(), 5);
	cudaDeviceReset();
	return 0;
}

