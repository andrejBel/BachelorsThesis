#include "KernelSharedMemory.h"

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

#include "GpuTimer.h"


using namespace std;




#define CONVOLUTIONSHARED(FILTER_W, BLOCK_S, TILE_S)\
case FILTER_W:\
{\
	float * ptr = (deviceFilters.get() + offset);\
	const int BLOCK_SIZE = BLOCK_S;\
	const int FILTER_WIDTH = FILTER_W;\
	const int TILE_SIZE = TILE_S;\
	static_assert(BLOCK_SIZE - TILE_SIZE >= (FILTER_WIDTH - 1), "Wrong block and tile size, BLOCKSIZE - TILESIZE >= (FILTERWIDTH - 1)");\
	const dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);\
	const dim3 gridSize((image.getNumCols() + TILE_SIZE - 1) / TILE_SIZE, (image.getNumRows() + TILE_SIZE - 1) / TILE_SIZE, 1);\
	timer.start(); \
	convolutionGPUShared<FILTER_WIDTH, BLOCK_SIZE, TILE_SIZE> << <gridSize, blockSize >> >(ptr, image.getNumRows(), image.getNumCols(), deviceGrayImageIn.get(), deviceGrayImageOut.get());\
	timer.stop(); \
	cout << "FilterWidth: " << FILTER_W << ", time: " << timer.getElapsedTime() << endl; \
	cout << "Bandwidth: " << (( ((gridSize.x * blockSize.x * gridSize.y * blockSize.y )  * sizeof(float) *(FILTER_W * FILTER_W + 1 + 1))  / ((timer.getElapsedTime()) * 1000000))) << " GB/s" << endl; \
	break;\
}

#define MERAJ(BLOCK_S,TILE_S,FILTER_W)\
{\
	cv::TickMeter m;\
	Filter< FILTER_W> * ptr = (Filter< FILTER_W> *) (deviceFilters.get() + offset);\
	const int BLOCK_SIZE = BLOCK_S;\
	const int FILTER_WIDTH = FILTER_W;\
	const int TILE_SIZE = TILE_S;\
	static_assert(BLOCK_SIZE - TILE_SIZE >= (FILTER_WIDTH - 1), "Wrong block and tile size, BLOCKSIZE - TILESIZE >= (FILTERWIDTH - 1)");\
	const dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);\
	const dim3 gridSize((image.getNumCols() + TILE_SIZE - 1) / TILE_SIZE, (image.getNumRows() + TILE_SIZE - 1) / TILE_SIZE, 1);\
	m.start();\
	convolutionGPUShared<FILTER_WIDTH, BLOCK_SIZE, TILE_SIZE> << <gridSize, blockSize >> >(ptr, image.getNumRows(), image.getNumCols(), deviceGrayImageIn.get(), deviceGrayImageOut.get());\
	m.stop();\
		checkCudaErrors(cudaDeviceSynchronize());\
	cout << "Block size: " << BLOCK_SIZE << ", TILE_SIZE: " << TILE_SIZE << ", FILTERWIDTH: " << FILTER_WIDTH << ", time: " << m.getTimeMicro() << endl;\
}

#define IMAD(a, b, c) ( __mul24((a), (b)) + (c) )

namespace processing
{

	template <typename R, typename int N>
	struct ForLoop
	{
		template< typename F>
		__device__ __forceinline__ R operator()(F f)
		{
			return f(N) + ForLoop<R,N - 1>()(f);
		}

	};

	template <typename R> 
	struct ForLoop<R,-1>
	{
		template<typename F>
		__device__ __forceinline__ R operator()(F f)
		{
			return 0;
		}

	};

	

	template<typename int FILTER_WIDTH, typename int BLOCK_SIZE, typename int TILE_SIZE>
	__global__ void convolutionGPUShared(float * filter, const int numRows, const int numCols, float * inputImage, float * outputImage)
	{
		
		int2 absoluteImagePosition;
		absoluteImagePosition.x = IMAD(blockIdx.x, TILE_SIZE, threadIdx.x);
		absoluteImagePosition.y = IMAD(blockIdx.y, TILE_SIZE, threadIdx.y);
		int2 sharedPosition;
		sharedPosition.x = absoluteImagePosition.x - (FILTER_WIDTH / 2);
		sharedPosition.y = absoluteImagePosition.y - (FILTER_WIDTH / 2);
		//__shared__ float filterShared[FILTER_WIDTH][FILTER_WIDTH];
		__shared__ float shared[BLOCK_SIZE][BLOCK_SIZE];
		int threadX = threadIdx.x;
		int threadY = threadIdx.y;
		sharedPosition.x = min(max(sharedPosition.x, 0), numCols - 1);
		sharedPosition.y = min(max(sharedPosition.y, 0), numRows - 1);
		shared[threadY][threadX] = inputImage[IMAD(sharedPosition.y, numCols, sharedPosition.x)];
		//if (threadX < FILTER_WIDTH && threadY < FILTER_WIDTH) 
		//{
			//filterShared[threadY][threadX] = filter[IMAD(threadY, FILTER_WIDTH, threadX)];
		//}
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
					result += filter[yOffset * FILTER_WIDTH +  xOffset] * shared[yOffset + threadY][xOffset + threadX];
				}
			}
			
			
			/*
			T result = ForLoop<T, FILTER_WIDTH - 1>()( [&result, &filterShared, &shared, threadX, threadY](int yOffset)
			{
				return ForLoop<T, FILTER_WIDTH - 1>()([&result, &filterShared, &shared, threadX, threadY, yOffset](int xOffset)
				{
					return filterShared[yOffset][xOffset] * shared[yOffset + threadY][xOffset + threadX];
				});
			});
			*/
			
			outputImage[IMAD(absoluteImagePosition.y, numCols, absoluteImagePosition.x)] = result;
		}
	}

	
	KernelSharedMemory::KernelSharedMemory() 
	{}

	
	void KernelSharedMemory::run(ImageFactory& image, vector<shared_ptr<Filter>>& filters, vector<shared_ptr<float>>& results)
	{
		GpuTimer timer;
		shared_ptr<float> deviceFilters = makeDeviceFilters(filters);
		// filter allocation and initialization
		shared_ptr<float> deviceGrayImageOut = allocateMemmoryDevice<float>(image.getNumPixels());
		const float* hostGrayImage = image.getInputGrayPointerFloat();

		shared_ptr<float> deviceGrayImageIn = allocateMemmoryDevice<float>(image.getNumPixels());
		checkCudaErrors(cudaMemcpy(deviceGrayImageIn.get(), hostGrayImage, image.getNumPixels() * sizeof(float), cudaMemcpyHostToDevice));
		// memory allocation

		uint offset = 0;
		for (auto& filter : filters)
		{
			switch (filter->getWidth())
			{
			CONVOLUTIONSHARED(1, 32, 32)
			CONVOLUTIONSHARED(3, 32, 30)
			CONVOLUTIONSHARED(5, 32, 28)
			CONVOLUTIONSHARED(7, 32, 26)
			CONVOLUTIONSHARED(9, 32, 24)
			CONVOLUTIONSHARED(11, 32, 22)
			CONVOLUTIONSHARED(13, 32, 20)
			CONVOLUTIONSHARED(15, 32, 18)
			default:
				std::cerr << "Filter with width: " << filter->getWidth() << " not supported!" << endl;
				break;
			}
			offset += filter->getSize();
			shared_ptr<float> resultCPU = makeArray<float>(image.getNumPixels());
			checkCudaErrors(cudaMemcpy(resultCPU.get(), deviceGrayImageOut.get(), image.getNumPixels() * sizeof(float), cudaMemcpyDeviceToHost));
			results.push_back(resultCPU);
		}
		cout << "";
	}

}

