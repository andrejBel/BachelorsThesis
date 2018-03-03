#include "KernelSharedMemory.h"

#include <cuda.h>
#include <helper_cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include "processing.h"
#include "Filter.h"

#include "opencv2/core/utility.hpp"


#include <vector>
#include <cstdio>
#include <cmath>
#include <iostream>

#include <thread>
#include <algorithm>
#include <type_traits>

using namespace std;


#define ALCHEMY_REPEAT_1(T)    T
#define ALCHEMY_REPEAT_2(T)    ALCHEMY_REPEAT_1(T) T
#define ALCHEMY_REPEAT_3(T)    ALCHEMY_REPEAT_2(T) T
#define ALCHEMY_REPEAT_4(T)    ALCHEMY_REPEAT_3(T) T
#define ALCHEMY_REPEAT_5(T)    ALCHEMY_REPEAT_4(T) T
#define ALCHEMY_REPEAT_6(T)    ALCHEMY_REPEAT_5(T) T
#define ALCHEMY_REPEAT_7(T)    ALCHEMY_REPEAT_6(T) T
#define ALCHEMY_REPEAT_8(T)    ALCHEMY_REPEAT_7(T) T
#define ALCHEMY_REPEAT_9(T)    ALCHEMY_REPEAT_8(T) T
#define ALCHEMY_REPEAT_10(T)    ALCHEMY_REPEAT_9(T) T
#define ALCHEMY_REPEAT_11(T)    ALCHEMY_REPEAT_10(T) T
#define ALCHEMY_REPEAT_12(T)    ALCHEMY_REPEAT_11(T) T
#define ALCHEMY_REPEAT_13(T)    ALCHEMY_REPEAT_12(T) T
#define ALCHEMY_REPEAT_14(T)    ALCHEMY_REPEAT_13(T) T
#define ALCHEMY_REPEAT_15(T)    ALCHEMY_REPEAT_14(T) T
#define ALCHEMY_REPEAT_16(T)    ALCHEMY_REPEAT_15(T) T
#define ALCHEMY_REPEAT_17(T)    ALCHEMY_REPEAT_16(T) T
#define ALCHEMY_REPEAT_18(T)    ALCHEMY_REPEAT_17(T) T
#define ALCHEMY_REPEAT_19(T)    ALCHEMY_REPEAT_18(T) T
#define ALCHEMY_REPEAT_20(T)    ALCHEMY_REPEAT_19(T) T
#define ALCHEMY_REPEAT_21(T)    ALCHEMY_REPEAT_20(T) T
#define ALCHEMY_REPEAT_22(T)    ALCHEMY_REPEAT_21(T) T
#define ALCHEMY_REPEAT_23(T)    ALCHEMY_REPEAT_22(T) T
#define ALCHEMY_REPEAT_24(T)    ALCHEMY_REPEAT_23(T) T
#define ALCHEMY_REPEAT_25(T)    ALCHEMY_REPEAT_24(T) T
#define ALCHEMY_REPEAT_26(T)    ALCHEMY_REPEAT_25(T) T
#define ALCHEMY_REPEAT_27(T)    ALCHEMY_REPEAT_26(T) T
#define ALCHEMY_REPEAT_28(T)    ALCHEMY_REPEAT_27(T) T
#define ALCHEMY_REPEAT_29(T)    ALCHEMY_REPEAT_28(T) T
#define ALCHEMY_REPEAT_30(T)    ALCHEMY_REPEAT_29(T) T
#define ALCHEMY_REPEAT_31(T)    ALCHEMY_REPEAT_30(T) T
#define ALCHEMY_REPEAT_32(T)   ALCHEMY_REPEAT_31(T) T

#define ALCHEMY_REPEAT_N(N,T)  ALCHEMY_REPEAT_##N(T)

#define CONVOLUTIONSHARED(FILTER_W, BLOCK_S, TILE_S)\
case FILTER_W:\
{\
	Filter<T, FILTER_W> * ptr = (Filter<T, FILTER_W> *) (deviceFilters.get() + offset);\
	const int BLOCK_SIZE = BLOCK_S;\
	const int FILTER_WIDTH = FILTER_W;\
	const int TILE_SIZE = TILE_S;\
	static_assert(BLOCK_SIZE - TILE_SIZE >= (FILTER_WIDTH - 1), "Wrong block and tile size, BLOCKSIZE - TILESIZE >= (FILTERWIDTH - 1)");\
	const dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);\
	const dim3 gridSize((image.getNumCols() + TILE_SIZE - 1) / TILE_SIZE, (image.getNumRows() + TILE_SIZE - 1) / TILE_SIZE, 1);\
	convolutionGPUShared<T, FILTER_WIDTH, BLOCK_SIZE, TILE_SIZE> << <gridSize, blockSize >> >(ptr, image.getNumRows(), image.getNumCols(), deviceGrayImageIn.get(), deviceGrayImageOut.get());\
	break;\
}

#define MERAJ(BLOCK_S,TILE_S,FILTER_W)\
{\
	cv::TickMeter m;\
	Filter<T, FILTER_W> * ptr = (Filter<T, FILTER_W> *) (deviceFilters.get() + offset);\
	const int BLOCK_SIZE = BLOCK_S;\
	const int FILTER_WIDTH = FILTER_W;\
	const int TILE_SIZE = TILE_S;\
	static_assert(BLOCK_SIZE - TILE_SIZE >= (FILTER_WIDTH - 1), "Wrong block and tile size, BLOCKSIZE - TILESIZE >= (FILTERWIDTH - 1)");\
	const dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);\
	const dim3 gridSize((image.getNumCols() + TILE_SIZE - 1) / TILE_SIZE, (image.getNumRows() + TILE_SIZE - 1) / TILE_SIZE, 1);\
	m.start();\
	convolutionGPUShared<T, FILTER_WIDTH, BLOCK_SIZE, TILE_SIZE> << <gridSize, blockSize >> >(ptr, image.getNumRows(), image.getNumCols(), deviceGrayImageIn.get(), deviceGrayImageOut.get());\
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

	

	template<typename T, typename int FILTER_WIDTH, typename int BLOCK_SIZE, typename int TILE_SIZE>
	__global__ void convolutionGPUShared(processing::Filter<T, FILTER_WIDTH> * filter, const int numRows, const int numCols, uchar * inputImage, T * outputImage)
	{
		
		int2 absoluteImagePosition;
		absoluteImagePosition.x = IMAD(blockIdx.x, TILE_SIZE, threadIdx.x);
		absoluteImagePosition.y = IMAD(blockIdx.y, TILE_SIZE, threadIdx.y);
		int2 sharedPosition;
		sharedPosition.x = absoluteImagePosition.x - (FILTER_WIDTH / 2);
		sharedPosition.y = absoluteImagePosition.y - (FILTER_WIDTH / 2);
		const T* filterV = filter->getFilter();
		__shared__ T filterShared[FILTER_WIDTH][FILTER_WIDTH];
		__shared__ uchar shared[BLOCK_SIZE][BLOCK_SIZE];
		int threadX = threadIdx.x;
		int threadY = threadIdx.y;
		sharedPosition.x = min(max(sharedPosition.x, 0), numCols - 1);
		sharedPosition.y = min(max(sharedPosition.y, 0), numRows - 1);
		shared[threadY][threadX] = inputImage[IMAD(sharedPosition.y, numCols, sharedPosition.x)];
		if (threadX < FILTER_WIDTH && threadY < FILTER_WIDTH) 
		{
			filterShared[threadY][threadX] = filterV[IMAD(threadY, FILTER_WIDTH, threadX)];
		}
		__syncthreads();
		if (threadX < TILE_SIZE && threadY < TILE_SIZE && absoluteImagePosition.x < numCols && absoluteImagePosition.y <  numRows)
		{
			
			T result(0.0);
#pragma unroll FILTER_WIDTH
			for (int yOffset = 0; yOffset < FILTER_WIDTH; yOffset++)
			{
#pragma unroll FILTER_WIDTH
				for (int xOffset = 0; xOffset < FILTER_WIDTH; xOffset++)
				{
					result += filterShared[yOffset][xOffset] * shared[yOffset + threadY][xOffset + threadX];
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

	template<typename T>
	KernelSharedMemory<T>::KernelSharedMemory() 
	{}

	template<typename T>
	void KernelSharedMemory<T>::run(ImageFactory& image, vector<shared_ptr<AbstractFilter<T>>>& filters, vector<shared_ptr<T>>& results)
	{
		uint filterCount(filters.size());
		size_t memmoryToAllocateForFiltersOnDevice(0);
		for_each(filters.begin(), filters.end(), [&memmoryToAllocateForFiltersOnDevice](auto& filter) { memmoryToAllocateForFiltersOnDevice += filter->getSize(); });
		shared_ptr<uchar> deviceFilters = allocateMemmoryDevice<uchar>(memmoryToAllocateForFiltersOnDevice);
		uint offset(0);
		int maxFilterWidth = 0;
		for_each(filters.begin(), filters.end(), [&deviceFilters, &offset, &maxFilterWidth](auto& filter)
		{
			filter->copyWholeFilterToDeviceMemory(deviceFilters.get() + offset);
			offset += filter->getSize();
			if (maxFilterWidth < filter->getSize())
			{
				maxFilterWidth = filter->getSize();
			}
		});
		// filter allocation and initialization
		shared_ptr<T> deviceGrayImageOut = allocateMemmoryDevice<T>(image.getNumPixels());
		uchar* hostGrayImage = image.getInputGrayPointer();

		shared_ptr<uchar> deviceGrayImageIn = allocateMemmoryDevice<uchar>(image.getNumPixels());
		checkCudaErrors(cudaMemcpy(deviceGrayImageIn.get(), hostGrayImage, image.getNumPixels() * sizeof(uchar), cudaMemcpyHostToDevice));
		// memory allocation

		offset = 0;
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
			shared_ptr<T> resultCPU = makeArray<T>(image.getNumPixels());
			checkCudaErrors(cudaMemcpy(resultCPU.get(), deviceGrayImageOut.get(), image.getNumPixels() * sizeof(T), cudaMemcpyDeviceToHost));
			results.push_back(resultCPU);
		}
		cout << "";
	}

}

