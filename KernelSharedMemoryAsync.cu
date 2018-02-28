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
#include <queue>
#include <mutex>

using namespace std;



namespace processing
{

	template<typename T, typename int FILTER_WIDTH, typename int BLOCK_SIZE, typename int TILE_SIZE>
	__global__ void convolutionGPUSharedAsync(processing::Filter<T, FILTER_WIDTH> * filter, const int numRows, const int numCols, uchar * inputImage, T * outputImage)
	{
		int2 absoluteImagePosition;
		absoluteImagePosition.x = blockIdx.x * TILE_SIZE + threadIdx.x;
		absoluteImagePosition.y = blockIdx.y * TILE_SIZE + threadIdx.y;
		int2 sharedPosition;
		sharedPosition.x = absoluteImagePosition.x - (FILTER_WIDTH / 2);
		sharedPosition.y = absoluteImagePosition.y - (FILTER_WIDTH / 2);
		__shared__ uchar shared[BLOCK_SIZE][BLOCK_SIZE];
		int threadX = threadIdx.x;
		int threadY = threadIdx.y;
		sharedPosition.x = min(max(sharedPosition.x, 0), numCols - 1);
		sharedPosition.y = min(max(sharedPosition.y, 0), numRows - 1);
		shared[threadY][threadX] = inputImage[sharedPosition.y * numCols + sharedPosition.x];
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
					result += filterV[yOffset*FILTER_WIDTH + xOffset] * shared[yOffset + threadY][xOffset + threadX];
				}
			}
			outputImage[absoluteImagePosition.y * numCols + absoluteImagePosition.x] = result;
		}
	}

	template<typename T>
	KernelSharedMemoryAsync<T>::KernelSharedMemoryAsync():
		threadPool_(THREADS_NUMBER)
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
		// filter allocation and initialization
		shared_ptr<T> deviceGrayImageOut = allocateMemmoryDevice<T>(image.getNumPixels());
		uchar* hostGrayImage = image.getInputGrayPointer();

		shared_ptr<uchar> deviceGrayImageIn = allocateMemmoryDevice<uchar>(image.getNumPixels());
		checkCudaErrors(cudaMemcpy(deviceGrayImageIn.get(), hostGrayImage, image.getNumPixels() * sizeof(uchar), cudaMemcpyHostToDevice));
		// memory allocation
		shared_ptr<T> result = allocateMemmoryDevice<T>(image.getNumPixels() * THREADS_NUMBER);

		size_t offsetForResult = 0;
		queue<T *> resultQueue;
		for (int i = 0; i < THREADS_NUMBER; i++)
		{
			resultQueue.push(result.get() + offsetForResult);
			offsetForResult += image.getNumPixels();
		}
		mutex mutex_;


		offset = 0;
		for (auto& filter : filters)
		{
			shared_ptr<T> result = allocateMemmoryDevice<T>(image.getNumPixels());
			switch (filter->getWidth())
			{
			case 3:
			{
				Filter<T, 3> * ptr = (Filter<T, 3> *) (deviceFilters.get() + offset);
				
				
				//threadPool_.addTask( [&mutex_, &resultQueue,&results, &image, ptr, deviceGrayImageIn] ()
			//	{	
					const int BLOCK_SIZE = 32;
					const int FILTER_WIDTH = 3;
					const int TILE_SIZE = 30;
					const dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
					const dim3 gridSize((image.getNumCols() + TILE_SIZE - 1) / TILE_SIZE, (image.getNumRows() + TILE_SIZE - 1) / TILE_SIZE, 1);
					mutex_.lock();
					T* outPut = resultQueue.front();
					resultQueue.pop();
					mutex_.unlock();
					convolutionGPUSharedAsync<T, FILTER_WIDTH, BLOCK_SIZE, TILE_SIZE> << <gridSize, blockSize >> >(ptr, image.getNumRows(), image.getNumCols(), deviceGrayImageIn.get(), outPut);
					checkCudaErrors(cudaDeviceSynchronize());
					cout << "preslo 1" << endl;
					shared_ptr<T> resultCpu = makeArray<T>(image.getNumPixels());
					checkCudaErrors(cudaMemcpy(resultCpu.get(), outPut, image.getNumPixels() * sizeof(T), cudaMemcpyDeviceToHost));
					cout << "preslo 2" << endl;
					checkCudaErrors(cudaDeviceSynchronize());
					cout << "preslo 3" << endl;
					mutex_.lock();
					results.push_back(resultCpu);
					resultQueue.push(outPut);
					mutex_.unlock();
				//}
				//);
				
				
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
				//convolutionGPUSharedAsync<T, FILTER_WIDTH, BLOCK_SIZE, TILE_SIZE> << <gridSize, blockSize >> > (ptr, image.getNumRows(), image.getNumCols(), deviceGrayImageIn.get(), result.get());
				//checkCudaErrors(cudaDeviceSynchronize());
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
				//convolutionGPUSharedAsync<T, FILTER_WIDTH, BLOCK_SIZE, TILE_SIZE> << <gridSize, blockSize >> >(ptr, image.getNumRows(), image.getNumCols(), deviceGrayImageIn.get(), result.get());

				//checkCudaErrors(cudaDeviceSynchronize());
				break;
			}
			default:
				break;
			}
			offset += filter->getSize();
			//results.push_back(result);
		}
		checkCudaErrors(cudaDeviceSynchronize());
		threadPool_.finishAll();
		cout << "";
	}

}

