#include "KernelSharedMemoryManaged.h"

#include <cuda.h>
#include <helper_cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include "opencv2/core/utility.hpp"

#include <vector>
#include <cstdio>
#include <cmath>
#include <iostream>

#include <thread>
#include <algorithm>

#define IMAD(a, b, c) ( __mul24((a), (b)) + (c) )
#define CONVOLUTIONGPUSHAREDMANAGED(FILTER_W, BLOCK_S, TILE_S)\
case FILTER_W:\
{\
	float * ptr =  (deviceFilters.get() + offset);\
	const int BLOCK_SIZE = BLOCK_S;\
	const int FILTER_WIDTH = FILTER_W;\
	const int TILE_SIZE = TILE_S;\
	static_assert(BLOCK_SIZE - TILE_SIZE >= (FILTER_WIDTH - 1), "Wrong block and tile size, BLOCKSIZE - TILESIZE >= (FILTERWIDTH - 1)");\
	const dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);\
	const dim3 gridSize((image.getNumCols() + TILE_SIZE - 1) / TILE_SIZE, (image.getNumRows() + TILE_SIZE - 1) / TILE_SIZE, 1);\
	convolutionGPUSharedManaged<FILTER_WIDTH, BLOCK_SIZE, TILE_SIZE> << <gridSize, blockSize >> >(ptr, image.getNumRows(), image.getNumCols(), deviceGrayImageIn.get(), resultUnified.get());\
	break;\
}

using namespace std;

namespace processing
{
	/*
	template<typename T, typename int FILTER_WIDTH, typename int BLOCK_SIZE, typename int TILE_SIZE>
	__global__ void convolutionGPUSharedManaged(processing::Filter<T, FILTER_WIDTH> * filter, const int numRows, const int numCols, uchar * inputImage, T * outputImage)
	{
	int2 absoluteImagePosition;
	absoluteImagePosition.x = IMAD(blockIdx.x, TILE_SIZE, threadIdx.x);
	absoluteImagePosition.y = IMAD(blockIdx.y, TILE_SIZE, threadIdx.y);
	int2 sharedPosition;
	sharedPosition.x = absoluteImagePosition.x - (FILTER_WIDTH / 2);
	sharedPosition.y = absoluteImagePosition.y - (FILTER_WIDTH / 2);
	__shared__ uchar shared[BLOCK_SIZE][BLOCK_SIZE];
	int threadX = threadIdx.x;
	int threadY = threadIdx.y;
	sharedPosition.x = min(max(sharedPosition.x, 0), numCols - 1);
	sharedPosition.y = min(max(sharedPosition.y, 0), numRows - 1);
	shared[threadY][threadX] = inputImage[IMAD(sharedPosition.y, numCols, sharedPosition.x)];
	__syncthreads();
	const T* filterV = filter->getFilter();
	T result(0.0);

	if (threadX < TILE_SIZE && threadY < TILE_SIZE && absoluteImagePosition.x < numCols && absoluteImagePosition.y <  numRows)
	{
	#pragma unroll FILTER_WIDTH
	for (int yOffset = 0; yOffset < FILTER_WIDTH; yOffset++)
	{
	#pragma unroll FILTER_WIDTH
	for (int xOffset = 0; xOffset < FILTER_WIDTH; xOffset++)
	{
	result += filterV[IMAD(yOffset, FILTER_WIDTH, xOffset)] * shared[yOffset + threadY][xOffset + threadX];
	}
	}
	outputImage[IMAD(absoluteImagePosition.y, numCols, absoluteImagePosition.x)] = result;
	}
	}
	*/

	template<typename int FILTER_WIDTH, typename int BLOCK_SIZE, typename int TILE_SIZE>
	__global__ void convolutionGPUSharedManaged(float * filter, const int numRows, const int numCols, float * inputImage, float * outputImage)
	{

		int2 absoluteImagePosition;
		absoluteImagePosition.x = IMAD(blockIdx.x, TILE_SIZE, threadIdx.x);
		absoluteImagePosition.y = IMAD(blockIdx.y, TILE_SIZE, threadIdx.y);
		int2 sharedPosition;
		sharedPosition.x = absoluteImagePosition.x - (FILTER_WIDTH / 2);
		sharedPosition.y = absoluteImagePosition.y - (FILTER_WIDTH / 2);
		__shared__ float filterShared[FILTER_WIDTH][FILTER_WIDTH];
		__shared__ float shared[BLOCK_SIZE][BLOCK_SIZE];
		int threadX = threadIdx.x;
		int threadY = threadIdx.y;
		sharedPosition.x = min(max(sharedPosition.x, 0), numCols - 1);
		sharedPosition.y = min(max(sharedPosition.y, 0), numRows - 1);
		shared[threadY][threadX] = inputImage[IMAD(sharedPosition.y, numCols, sharedPosition.x)];
		if (threadX < FILTER_WIDTH && threadY < FILTER_WIDTH)
		{
			filterShared[threadY][threadX] = filter[IMAD(threadY, FILTER_WIDTH, threadX)];
		}
		__syncthreads();
		if (threadX < TILE_SIZE && threadY < TILE_SIZE && absoluteImagePosition.x < numCols && absoluteImagePosition.y <  numRows)
		{

			float result(0.0);
#pragma unroll FILTER_WIDTH
			for (int yOffset = 0; yOffset < FILTER_WIDTH; yOffset++)
			{
#pragma unroll FILTER_WIDTH
				for (int xOffset = 0; xOffset < FILTER_WIDTH; xOffset++)
				{
					result += filterShared[yOffset][xOffset] * shared[yOffset + threadY][xOffset + threadX];
				}
			}
			outputImage[IMAD(absoluteImagePosition.y, numCols, absoluteImagePosition.x)] = result;
		}
	}


	KernelSharedMemoryManaged::KernelSharedMemoryManaged()
	{}


	void KernelSharedMemoryManaged::run(ImageFactory& image, vector<shared_ptr<Filter>>& filters, vector<shared_ptr<float>>& results)
	{
		shared_ptr<float> deviceFilters = makeDeviceFilters(filters);

		// filter allocation and initialization
		float* hostGrayImage = image.getInputGrayPointerFloat();

		shared_ptr<float> deviceGrayImageIn = allocateMemmoryDevice<float>(image.getNumPixels());

		checkCudaErrors(cudaMemcpy(deviceGrayImageIn.get(), hostGrayImage, image.getNumPixels() * sizeof(float), cudaMemcpyHostToDevice));
		// memory allocation
		shared_ptr<float> deviceGrayImageOut = allocateMemmoryDevice<float>(image.getNumPixels());
		uint offset = 0;
		for (auto& filter : filters)
		{
			shared_ptr<float> resultUnified = allocateManagedMemory<float>(image.getNumPixels());
			checkCudaErrors(cudaDeviceSynchronize());
			switch (filter->getWidth())
			{
				CONVOLUTIONGPUSHAREDMANAGED(1, 32, 32)
					CONVOLUTIONGPUSHAREDMANAGED(3, 32, 30)
					CONVOLUTIONGPUSHAREDMANAGED(5, 32, 28)
					CONVOLUTIONGPUSHAREDMANAGED(7, 32, 26)
					CONVOLUTIONGPUSHAREDMANAGED(9, 32, 24)
					CONVOLUTIONGPUSHAREDMANAGED(11, 32, 22)
					CONVOLUTIONGPUSHAREDMANAGED(13, 32, 20)
					CONVOLUTIONGPUSHAREDMANAGED(15, 32, 18)
			default:
				std::cerr << "Filter with width: " << filter->getWidth() << " not supported!" << endl;
				break;
			}
			offset += filter->getSize();
			//checkCudaErrors(cudaMemcpy(resultCPU.get(), deviceGrayImageOut.get(), image.getNumPixels() * sizeof(T), cudaMemcpyDeviceToHost));
			results.push_back(resultUnified);
		}
		checkCudaErrors(cudaDeviceSynchronize());
	}

}

