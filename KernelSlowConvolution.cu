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

using namespace std;



namespace processing
{

template <typename T>
__device__ const T& min(const T& a, const T& b) {
	return !(b<a) ? a : b;
}

template<typename T>
__global__ void kernel(uchar * inputImage, uchar * outputImage,T** filters,uint* filterWidths, uint* filterHeights, uint numberOfFilters, const size_t numRows, const size_t numCols)
{
	const auto absolute_image_position_x = blockIdx.x * blockDim.x + threadIdx.x;
	const auto absolute_image_position_y = blockIdx.y * blockDim.y + threadIdx.y;
	if (absolute_image_position_x >= numCols || absolute_image_position_y >= numRows)
	{
		return;
	}
	const auto index1D = absolute_image_position_y * numCols + absolute_image_position_x;
	auto point1D(index1D);
	T* filter = nullptr;
	uint filterWidth = 0;
	int filterDifferenceWidth = 0;
	int filterDifferenceHeight = 0;
	T result(0.0);
	if (index1D == 0)
	{
		for (uint i = 0; i < numberOfFilters; ++i)
		{
			filter = filters[i];
			auto filterWidth = filterWidths[i];
			auto filterHeight = filterHeights[i];
			filterDifferenceWidth = filterWidths[i] / 2;
			filterDifferenceHeight = filterHeights[i] / 2;

			printf("Filter width: %d\n", filterWidth);
			printf("Filter height: %d\n", filterHeight);
			printf("filterDifferenceWidth: %d\n", filterDifferenceWidth);
			printf("filterDifferenceHeight: %d\n", filterDifferenceHeight);

			for (uint j = 0; j < filterHeight * filterWidth; j++)
			{
				printf("Filter position: %d, value: %f\n", i, filter[j]);
			}
			printf("Multiplier value: %f\n", filter[filterWidth * filterHeight]);
			for (int j = -filterDifferenceWidth; j <= filterDifferenceWidth; j++)
			{
				for (int k = -filterDifferenceHeight; k <= filterDifferenceHeight; k++)
				{
					point1D = processing::min<size_t>(max(0, absolute_image_position_y + k), numRows - 1) * numCols + processing::min<size_t>(max(0, absolute_image_position_x + j), numCols - 1);
					result += filter[(j + filterDifferenceHeight / 2)*filterWidth + k + filterDifferenceWidth] * inputImage[point1D];
					printf("Result value: %f\n", result);
				}
			}
			outputImage[index1D] = result;

		}
	}
	for (uint i = 0; i < numberOfFilters; ++i)
	{
		filter = filters[i];
		filterWidth = filterWidths[i];
		filterDifferenceWidth = filterWidths[i] / 2;
		filterDifferenceHeight = filterHeights[i] / 2;


		for (int j = -filterDifferenceWidth; j <= filterDifferenceWidth; j++)
		{
			for (int k = -filterDifferenceHeight; k <= filterDifferenceHeight; k++)
			{
				point1D = processing::min<size_t>(max(0, absolute_image_position_y + k), numRows - 1) * numCols + processing::min<size_t>(::max(0, absolute_image_position_x + j), numCols - 1);
				result += filter[(j + filterDifferenceHeight / 2)*filterWidth + k + filterDifferenceWidth] * inputImage[point1D];
			}
		}
		printf("Result value: %d\n", static_cast<uchar>(floor(result)));
		outputImage[index1D] = 50;
		//outputImage[index1D] = static_cast<uchar>(floor(result));
		//__syncthreads();
		//inputImage[index1D] = outputImage[index1D];

	}	
}



	template<typename T>
	KernelSlowConvolution<T>::KernelSlowConvolution(vector<Filter<T>>& filters):
		h_filters_(filters)
	{
	}

	template<typename T>
	void KernelSlowConvolution<T>::run(ImageFactory & image)
	{
		auto filterCount(h_filters_.size());
		vector<T *> hostPointersFilterValues(filterCount);
		vector<uint> hostFilterWidth(filterCount);
		vector<uint> hostFilterHeight(filterCount);

		auto deviceFilters = allocateMemmoryDevice<T *>(filterCount);
		auto deviceFilterWidth = allocateMemmoryDevice<uint>(filterCount);
		auto deviceFilterHeight = allocateMemmoryDevice<uint>(filterCount);

		for (auto i = 0; i < filterCount; ++i)
		{
			h_filters_[i].allocateAndCopyHostFilterToDevice();
			hostPointersFilterValues[i] = h_filters_[i].getDeviceFilterPointer();
			hostFilterWidth[i] = h_filters_[i].getFilterWidth();
			hostFilterHeight[i] = h_filters_[i].getFilterHeight();
		}

		const uint numberOfThreadsInBlock = 32;
		const dim3 blockSize(numberOfThreadsInBlock, numberOfThreadsInBlock);
		const dim3 gridSize((image.getNumCols() + blockSize.x - 1) / blockSize.x, (image.getNumRows() + blockSize.y - 1) / blockSize.y, 1);

		auto outputImage = allocateMemmoryDevice<uchar>(image.getNumRows() * image.getNumCols());
		
		checkCudaErrors(cudaMemcpy(deviceFilters, hostPointersFilterValues.data(), filterCount * sizeof(T *), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(deviceFilterWidth, hostFilterWidth.data(), filterCount * sizeof(uint), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(deviceFilterHeight, hostFilterHeight.data(), filterCount * sizeof(uint), cudaMemcpyHostToDevice));

		kernel<T><<<1, 1 >>>(image.getDeviceGrayPointer(), outputImage, deviceFilters,deviceFilterWidth, deviceFilterHeight, filterCount, image.getNumRows(), image.getNumCols());


		image.copyDeviceGrayToHostGray(outputImage);
		//for (size_t i = 0; i < image.getNumRows() * image.getNumCols(); i++)
		//{
			printf("%d \n",image.getHostGrayPointer()[0]);
		//}
		for (auto i = 0; i < filterCount; ++i)
		{
			h_filters_[i].deallocateDeviceFilter();
		}
		deallocateMemmoryDevice(outputImage);
		deallocateMemmoryDevice(deviceFilterWidth);
		deallocateMemmoryDevice(deviceFilterHeight);
	}

}


