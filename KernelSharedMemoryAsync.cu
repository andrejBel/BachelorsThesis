#include "KernelSharedMemoryAsync.h"

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
#include "ThreadPool.h"

using namespace std;


#define CONVOLUTIONSHAREDASYNC(FILTER_W, BLOCK_S, TILE_S)\
case FILTER_W:\
{\
	Filter<T, FILTER_W> * ptr = (Filter<T, FILTER_W> *) (deviceFilters.get() + offset);\
	const int BLOCK_SIZE = BLOCK_S;\
	const int FILTER_WIDTH = FILTER_W;\
	const int TILE_SIZE = TILE_S;\
	static_assert(BLOCK_SIZE - TILE_SIZE >= (FILTER_WIDTH - 1), "Wrong block and tile size, BLOCKSIZE - TILESIZE >= (FILTERWIDTH - 1)");\
	const dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);\
	const dim3 gridSize((image.getNumCols() + TILE_SIZE - 1) / TILE_SIZE, (image.getNumRows() + TILE_SIZE - 1) / TILE_SIZE, 1);\
	convolutionGPUSharedAsync<T, FILTER_WIDTH, BLOCK_SIZE, TILE_SIZE> << <gridSize, blockSize >> >(ptr, image.getNumRows(), image.getNumCols(), deviceGrayImageIn.get(), deviceGrayImageOut.get());\
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

	template<typename T, typename int FILTER_WIDTH, typename int BLOCK_SIZE, typename int TILE_SIZE>
	__global__ void convolutionGPUSharedAsync(processing::Filter<T, FILTER_WIDTH> * filter, const int numRows, const int numCols,int xOffset, int yOffset , uchar * inputImage, T * outputImage)
	{

		int2 absoluteImagePosition;
		absoluteImagePosition.x = blockIdx.x * TILE_SIZE + threadIdx.x + xOffset;
		absoluteImagePosition.y = blockIdx.y *  TILE_SIZE + threadIdx.y + yOffset;
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
			outputImage[IMAD(absoluteImagePosition.y, numCols, absoluteImagePosition.x)] = result;
		}
	}

	template<typename T>
	KernelSharedMemoryAsync<T>::KernelSharedMemoryAsync()
	{}

	template<typename T>
	void KernelSharedMemoryAsync<T>::run(ImageFactory& image, vector<shared_ptr<AbstractFilter<T>>>& filters, vector<shared_ptr<T>>& results)
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
		offset = 0;
		// filter allocation and initialization
		const int numberOfStreams = 2;
		CudaStream  streams[numberOfStreams];
		ThreadPool threadPools[numberOfStreams];
		// stream initialization
		shared_ptr<T> deviceGrayImageOut = allocateMemmoryDevice<T>(image.getNumPixels());
		uchar* hostGrayImage = image.getInputGrayPointer();

		shared_ptr<uchar> deviceGrayImageIn = allocateMemmoryDevice<uchar>(image.getNumPixels());
		checkCudaErrors(cudaMemcpy(deviceGrayImageIn.get(), hostGrayImage, image.getNumPixels() * sizeof(uchar), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaDeviceSynchronize());
		// memory allocation
		size_t pixels = image.getNumPixels(); 
		int numberOfColumns = image.getNumCols(); // x
		int numberOfRows = image.getNumRows(); // y
		const int coeficientsForOffset[numberOfStreams][2] = { {0,0}, {1, 1} };
		for (auto& filter : filters)
		{
			shared_ptr<T> resultCpu = makeArray<T>(pixels);
			checkCudaErrors(cudaHostRegister(resultCpu.get(), pixels * sizeof(T), cudaHostRegisterPortable));
			checkCudaErrors(cudaDeviceSynchronize());
			switch (filter->getWidth())
			{
			
			
			default:
				std::cerr << "Filter with width: " << filter->getWidth() << " not supported!" << endl;
				break;
			}
			offset += filter->getSize();
			checkCudaErrors(cudaHostUnregister(resultCpu.get()));
			results.push_back(resultCpu);
		}
		cout << "";
	}

}

