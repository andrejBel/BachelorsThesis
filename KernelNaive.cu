#include "KernelNaive.h"

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

#define CONVOLUTIONSLOWNAIVE(FILTERWIDTH)\
case FILTERWIDTH:\
{\
	Filter<T, FILTERWIDTH> * ptr = (Filter<T, FILTERWIDTH> *) (deviceFilters.get() + offset);\
	convolutionGPUNaive << <gridSize, blockSize >> >(ptr, image.getNumRows(), image.getNumCols(), deviceGrayImageIn.get(), deviceGrayImageOut.get(), FILTERWIDTH);\
	break;\
}

	template<typename T, typename int FILTER_WIDTH>
	__global__ void convolutionGPUNaive(processing::Filter<T, FILTER_WIDTH> * filter, const int numRows, const int numCols, uchar * inputImage, T * outputImage, int filterWidth)
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
		for (int yOffset = 0; yOffset < filterWidth; yOffset++)
		{
			for (int xOffset = 0; xOffset < filterWidth; xOffset++)
			{
				pointPosition.x = absoluteImagePosition.x + xOffset - filterWidth / 2;
				pointPosition.y = absoluteImagePosition.y + yOffset - filterWidth / 2;
				pointPosition.x = KernelNaive<T>::min(KernelNaive<T>::max(pointPosition.x, 0), numCols - 1);
				pointPosition.y = KernelNaive<T>::min(KernelNaive<T>::max(pointPosition.y, 0), numRows - 1);
				result += filterV[yOffset*filterWidth + xOffset] * inputImage[pointPosition.y*numCols + pointPosition.x];
			}
		}
		outputImage[index1D] = result;
	}

	template<typename T>
	KernelNaive<T>::KernelNaive()
	{
	}

	template<typename T>
	void KernelNaive<T>::run(ImageFactory& image, vector<shared_ptr<AbstractFilter<T>>>& filters, vector<shared_ptr<T>>& results)
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
				CONVOLUTIONSLOWNAIVE(1)
				CONVOLUTIONSLOWNAIVE(3)
				CONVOLUTIONSLOWNAIVE(5)
				CONVOLUTIONSLOWNAIVE(7)
				CONVOLUTIONSLOWNAIVE(9)
				CONVOLUTIONSLOWNAIVE(11)
				CONVOLUTIONSLOWNAIVE(13)
				CONVOLUTIONSLOWNAIVE(15)
			default:
				std::cerr << "Filter with width: " << filter->getWidth() << " not supported!" << endl;
				break;
			}
			offset += filter->getSize();
			shared_ptr<T> resultCPU = makeArray<T>(image.getNumPixels());
			checkCudaErrors(cudaMemcpy(resultCPU.get(), deviceGrayImageOut.get(), image.getNumPixels() * sizeof(T), cudaMemcpyDeviceToHost));
			results.push_back(resultCPU);
		}
		checkCudaErrors(cudaDeviceSynchronize());
	}

}
