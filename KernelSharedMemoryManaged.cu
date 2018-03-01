#include "KernelSharedMemoryManaged.h"

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

#define IMAD(a, b, c) ( __mul24((a), (b)) + (c) )

using namespace std;

namespace processing
{

	template<typename T, typename int FILTER_WIDTH, typename int BLOCK_SIZE, typename int TILE_SIZE>
	__global__ void convolutionGPUSharedManaged(processing::Filter<T, FILTER_WIDTH> * filter, const int numRows, const int numCols, uchar * inputImage, T * outputImage)
	{
		int2 absoluteImagePosition;
		absoluteImagePosition.x = IMAD(blockIdx.x , TILE_SIZE ,threadIdx.x);
		absoluteImagePosition.y = IMAD(blockIdx.y , TILE_SIZE, threadIdx.y);
		int2 sharedPosition;
		sharedPosition.x = absoluteImagePosition.x - (FILTER_WIDTH / 2);
		sharedPosition.y = absoluteImagePosition.y - (FILTER_WIDTH / 2);
		__shared__ uchar shared[BLOCK_SIZE][BLOCK_SIZE];
		int threadX = threadIdx.x;
		int threadY = threadIdx.y;
		sharedPosition.x = min(max(sharedPosition.x, 0), numCols - 1);
		sharedPosition.y = min(max(sharedPosition.y, 0), numRows - 1);
		shared[threadY][threadX] = inputImage[IMAD(sharedPosition.y , numCols , sharedPosition.x)];
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
					result += filterV[IMAD(yOffset,FILTER_WIDTH , xOffset)] * shared[yOffset + threadY][xOffset + threadX];
				}
			}
			outputImage[IMAD(absoluteImagePosition.y, numCols ,absoluteImagePosition.x)] = result;
		}
	}

	template<typename T>
	KernelSharedMemoryManaged<T>::KernelSharedMemoryManaged()
	{}

	template<typename T>
	void KernelSharedMemoryManaged<T>::run(ImageFactory& image, vector<shared_ptr<AbstractFilter<T>>>& filters, vector<shared_ptr<T>>& results)
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
		uchar* hostGrayImage = image.getInputGrayPointer();

		shared_ptr<uchar> deviceGrayImageIn = allocateMemmoryDevice<uchar>(image.getNumPixels());
		//std::copy(hostGrayImage, hostGrayImage + image.getNumRows(), deviceGrayImageIn.get());
		checkCudaErrors(cudaMemcpy(deviceGrayImageIn.get(), hostGrayImage, image.getNumPixels() * sizeof(uchar), cudaMemcpyHostToDevice));
		// memory allocation

		for (auto& filter : filters)
		{
			shared_ptr<T> resultUnified = allocateManagedMemory<T>(image.getNumPixels());
			//cudaMemPrefetchAsync(resultUnified.get(), image.getNumPixels() * sizeof(T), device, NULL);
			switch (filter->getWidth())
			{
			case 3:
			{
				Filter<T, 3> * ptr = (Filter<T, 3> *) (deviceFilters.get() + offset);
				const int BLOCK_SIZE = 32;
				const int FILTER_WIDTH = 3;
				const int TILE_SIZE = 30;
				const dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
				const dim3 gridSize((image.getNumCols() + TILE_SIZE - 1) / TILE_SIZE, (image.getNumRows() + TILE_SIZE - 1) / TILE_SIZE, 1);
				convolutionGPUSharedManaged<T, FILTER_WIDTH, BLOCK_SIZE, TILE_SIZE> << <gridSize, blockSize >> >(ptr, image.getNumRows(), image.getNumCols(), deviceGrayImageIn.get(), resultUnified.get());
				break;
			}
			case 5:
			{
				Filter<T, 5> * ptr = (Filter<T, 5> *) (deviceFilters.get() + offset);
				const int BLOCK_SIZE = 32;
				const int FILTER_WIDTH = 5;
				const int TILE_SIZE = 28;
				const dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
				const dim3 gridSize((image.getNumCols() + TILE_SIZE - 1) / TILE_SIZE, (image.getNumRows() + TILE_SIZE - 1) / TILE_SIZE, 1);
				static_assert(BLOCK_SIZE - TILE_SIZE >= (FILTER_WIDTH - 1), "Wrong block and tile size, BLOCKSIZE - TILESIZE >= (FILTERWIDTH - 1)");
				convolutionGPUSharedManaged<T, FILTER_WIDTH, BLOCK_SIZE, TILE_SIZE> << <gridSize, blockSize >> > (ptr, image.getNumRows(), image.getNumCols(), deviceGrayImageIn.get(), resultUnified.get());
				break;
			}
			case 7:
			{
				Filter<T, 7> * ptr = (Filter<T, 7> *) (deviceFilters.get() + offset);
				const int BLOCK_SIZE = 32;
				const int FILTER_WIDTH = 7;
				const int TILE_SIZE = 26;
				const dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
				const dim3 gridSize((image.getNumCols() + TILE_SIZE - 1) / TILE_SIZE, (image.getNumRows() + TILE_SIZE - 1) / TILE_SIZE, 1);
				convolutionGPUSharedManaged<T, FILTER_WIDTH, BLOCK_SIZE, TILE_SIZE> << <gridSize, blockSize >> >(ptr, image.getNumRows(), image.getNumCols(), deviceGrayImageIn.get(), resultUnified.get());
				break;
			}
			default:
				break;
			}
			offset += filter->getSize();
			//checkCudaErrors(cudaMemcpy(resultCPU.get(), deviceGrayImageOut.get(), image.getNumPixels() * sizeof(T), cudaMemcpyDeviceToHost));
			
			results.push_back(resultUnified);
		}
		checkCudaErrors(cudaDeviceSynchronize());
		
	}

}

