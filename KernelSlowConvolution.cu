#include "KernelSlowConvolution.h"

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
using namespace std;



namespace processing
{

template <typename T>
__device__ const T& min(const T& a, const T& b) {
	return !(b<a) ? a : b;
}

template <typename T>
__device__ const T& max(const T& a, const T& b) {
	return (b<a) ? a : b;
}



template<typename T,typename int WIDTH>
__global__ void fun(processing::Filter<T,WIDTH> * filter, const int numRows, const int numCols, T * outputImage, uchar * inputImage)
{
	int2 absoluteImagePosition;
	
	absoluteImagePosition.x = blockIdx.x * blockDim.x + threadIdx.x;
	absoluteImagePosition.y = blockIdx.y * blockDim.y + threadIdx.y;
	if (absoluteImagePosition.x >= numCols || absoluteImagePosition.y >= numRows)
	{
		return;
	}
	const size_t index1D = absoluteImagePosition.y * numCols + absoluteImagePosition.x;
	const T* filterV = filter->getFilter();
	T result(0.0);
	int2 pointPosition;

	#pragma unroll
	for (int yOffset = 0; yOffset < WIDTH; yOffset++)
	{
		#pragma unroll
		for (int xOffset = 0; xOffset < WIDTH; xOffset++)
		{

			pointPosition.x = absoluteImagePosition.x + xOffset - WIDTH / 2;
			pointPosition.y = absoluteImagePosition.y + yOffset - WIDTH / 2;
			pointPosition.x = min(max(pointPosition.x, 0), numCols - 1);
			pointPosition.y = min(max(pointPosition.y, 0), numRows - 1);
			
			
			result += filterV[yOffset*WIDTH + xOffset] * inputImage[ pointPosition.y * numCols + pointPosition.x];
			//point1D = processing::min<size_t>(max(0, absolute_image_position_y + k), numRows - 1) * numCols + processing::min<size_t>(max(0, absolute_image_position_x + j), numCols - 1);
			//result += filter[(j + filterDifferenceHeight / 2)*filterWidth + k + filterDifferenceWidth] * inputImage[point1D];
			//printf("Result value: %f\n", result);
		}
	}
	//printf("Result: %f\n", result);
	outputImage[index1D] = result;
}


	template<typename T>
	KernelSlowConvolution<T>::KernelSlowConvolution(vector<shared_ptr<AbstractFilter<T>>>& filters):
		h_filters_(filters)
	{
	}

	template<typename T>
	void KernelSlowConvolution<T>::run(ImageFactory & image)
	{
		uint filterCount(h_filters_.size());
		size_t memmoryToAllocateForFiltersOnDevice(0);
		for_each(h_filters_.begin(), h_filters_.end(), [&memmoryToAllocateForFiltersOnDevice](auto& filter) { memmoryToAllocateForFiltersOnDevice += filter->getSize(); });
		
		uchar* deviceFilters = allocateMemmoryDevice<uchar>( memmoryToAllocateForFiltersOnDevice);

		uint offset(0);
		for_each(h_filters_.begin(), h_filters_.end(), [&deviceFilters, &offset](auto& filter)
		{ 
			filter->copyWholeFilterToDeviceMemory(deviceFilters + offset);
			offset += filter->getSize();
		});

		const uint numberOfThreadsInBlock = 32;
		const dim3 blockSize(numberOfThreadsInBlock, numberOfThreadsInBlock);
		const dim3 gridSize((image.getNumCols() + blockSize.x - 1) / blockSize.x, (image.getNumRows() + blockSize.y - 1) / blockSize.y, 1);


		auto outputImage = allocateMemmoryDevice<T>(image.getNumRows() * image.getNumCols());
		
		//kernel<T><<<1, 1 >>>(image.getDeviceGrayPointer(), outputImage, deviceFilters,deviceFilterWidth, deviceFilterHeight, filterCount, image.getNumRows(), image.getNumCols());
		offset = 0;
		for (auto& filter : h_filters_)
		{
			switch (filter->getWidth())
			{
			case 3: 
			{
				Filter<T, 3> * ptr = (Filter<T, 3> *) (deviceFilters + offset);
				fun<<<gridSize, blockSize>>>(ptr, image.getNumRows(), image.getNumCols(), outputImage, image.getDeviceGrayPointer());
				break;
			}
			case 5:
			{
				Filter<T, 5> * ptr = (Filter<T, 5> *) (deviceFilters + offset);
				fun << <gridSize, blockSize >> >(ptr, image.getNumRows(), image.getNumCols(), outputImage, image.getDeviceGrayPointer());
				break;
			}
			case 7:
			{
				Filter<T, 7> * ptr = (Filter<T, 7> *) (deviceFilters + offset);
				fun << <gridSize, blockSize >> >(ptr, image.getNumRows(), image.getNumCols(), outputImage, image.getDeviceGrayPointer());
				break;
			}
			default:
				break;
			}
			offset += filter->getSize();
		}

		T * outT = (T *)malloc(image.getNumRows()* image.getNumCols() * sizeof(T));
		checkCudaErrors(cudaMemcpy(outT,outputImage, image.getNumRows()* image.getNumCols() * sizeof(T), cudaMemcpyDeviceToHost));
		uchar * outC = image.getHostGrayPointer();

		T min = 1000.0;
		T max = 0.0;
		for (size_t i = 0; i < image.getNumRows()* image.getNumCols(); i++)
		{
			if (min > outT[i]) 
			{
				min = outT[i];
			}
			if (max < outT[i])
			{
				max = outT[i];
			}
		}
		T diff = 255.0 / (max - min);
		for (size_t i = 0; i < image.getNumRows()* image.getNumCols(); i++) 
		{
			outC[i] = uchar(floor(diff * outT[i]));
			//outC[i] = uchar(floor( outT[i]));
		}
		deallocateMemmoryDevice(deviceFilters);
		deallocateMemmoryDevice(outputImage);
		free(outT);
	}

}


