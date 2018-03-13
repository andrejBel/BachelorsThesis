#include "KernelSlowConvolutionNoEdgeCopy.h"

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

using namespace std;




namespace processing
{

#define CONVOLUTIONSLOWNOEDGECOPY(FILTERWIDTH)\
case FILTERWIDTH:\
{\
	Filter<T, FILTERWIDTH> * ptr = (Filter<T, FILTERWIDTH> *) (deviceFilters.get() + offset);\
	convolutionGPU << <gridSize, blockSize >> >(ptr, image.getNumRows(), image.getNumCols(), deviceGrayImageIn.get(), deviceGrayImageOut.get());\
	break;\
}

	template<typename T, typename int FILTER_WIDTH>
	__global__ void convolutionGPU(processing::Filter<T, FILTER_WIDTH> * filter, const int numRows, const int numCols, uchar * inputImage, T * outputImage)
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
		#pragma unroll FILTER_WIDTH
		for (int yOffset = 0; yOffset < FILTER_WIDTH; yOffset++)
		{
		#pragma unroll FILTER_WIDTH
			for (int xOffset = 0; xOffset < FILTER_WIDTH; xOffset++)
			{
				pointPosition.x = absoluteImagePosition.x + xOffset - FILTER_WIDTH / 2;
				pointPosition.y = absoluteImagePosition.y + yOffset - FILTER_WIDTH / 2;
				pointPosition.x = min(max(pointPosition.x, 0), numCols - 1);
				pointPosition.y = min(max(pointPosition.y, 0), numRows - 1);
				result += filterV[yOffset*FILTER_WIDTH + xOffset] * inputImage[pointPosition.y*numCols + pointPosition.x];
			}
		}
		outputImage[index1D] = result;

	}


	template<typename T>
	__host__ __forceinline__ int KernelSlowConvolutionNoEdgeCopy<T>::indexToCopyToMirrored(int index, int numCols, int numRows, const int filterWidth)
	{
		int indexX = (index % (numCols + (filterWidth / 2) * 2)) - (filterWidth / 2);
		int indexY = (index / (numCols + (filterWidth / 2) * 2)) - (filterWidth / 2);
		indexX = std::min(std::max(indexX, 0), numCols - 1);
		indexY = std::min(std::max(indexY, 0), numRows - 1);
		return indexY * numCols + indexX;
	}



	template<typename T>
	KernelSlowConvolutionNoEdgeCopy<T>::KernelSlowConvolutionNoEdgeCopy() 
	{
	}

	template<typename T>
	void KernelSlowConvolutionNoEdgeCopy<T>::run(ImageFactory& image, vector<shared_ptr<AbstractFilter<T>>>& filters, vector<shared_ptr<T>>& results)
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

		const uint numberOfThreadsInBlock = 16;
		const dim3 blockSize(numberOfThreadsInBlock, numberOfThreadsInBlock);
		const dim3 gridSize((image.getNumCols() + blockSize.x - 1) / blockSize.x, (image.getNumRows() + blockSize.y - 1) / blockSize.y, 1);
		// kernels parameters
		offset = 0;
		for (auto& filter : filters)
		{
			switch (filter->getWidth())
			{
			CONVOLUTIONSLOWNOEDGECOPY(1)
			CONVOLUTIONSLOWNOEDGECOPY(3)
			CONVOLUTIONSLOWNOEDGECOPY(5)
			CONVOLUTIONSLOWNOEDGECOPY(7)
			CONVOLUTIONSLOWNOEDGECOPY(9)
			CONVOLUTIONSLOWNOEDGECOPY(11)
			CONVOLUTIONSLOWNOEDGECOPY(13)
			CONVOLUTIONSLOWNOEDGECOPY(15)
			default:
				std::cerr << "Filter with width: " << filter->getWidth() << " not supported!" << endl;
				break;
			}
			offset += filter->getSize();
			shared_ptr<T> resultCPU = makeArray<T>(image.getNumPixels());
			checkCudaErrors(cudaDeviceSynchronize());
			checkCudaErrors(cudaMemcpy(resultCPU.get(), deviceGrayImageOut.get(), image.getNumPixels() * sizeof(T), cudaMemcpyDeviceToHost));
			results.push_back(resultCPU);		
		}
	}
	
}
