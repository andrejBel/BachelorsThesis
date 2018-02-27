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

template <typename T>
__global__ void separateChannels(const uchar4* const inputImageRGBA, int numRows, int numCols, unsigned char* const redChannel, unsigned char* const greenChannel, unsigned char* const blueChannel)
{
	int absolute_image_position_x = blockIdx.x * blockDim.x + threadIdx.x;
	int absolute_image_position_y = blockIdx.y * blockDim.y + threadIdx.y;
	if (absolute_image_position_x >= numCols || absolute_image_position_y >= numRows)
	{
		return;
	}
	const int index_1D = absolute_image_position_y * numCols + absolute_image_position_x;
	redChannel[index_1D] = inputImageRGBA[index_1D].x;
	greenChannel[index_1D] = inputImageRGBA[index_1D].y;
	blueChannel[index_1D] = inputImageRGBA[index_1D].z;
}

template <typename T>
__global__ void recombineChannels(const unsigned char* const redChannel, const unsigned char* const greenChannel, const unsigned char* const blueChannel, uchar4* const outputImageRGBA, int numRows, int numCols)
{
	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);

	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
		return;
	outputImageRGBA[thread_1D_pos] = make_uchar4(redChannel[thread_1D_pos], greenChannel[thread_1D_pos], blueChannel[thread_1D_pos], 255);
}

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

	__device__ __forceinline__ size_t indexInNew(int indexX, int indexY, int originalWidth, int originalHeight, int filterWidth)
	{
		int newWidth = originalWidth + (filterWidth / 2) * 2;
		indexX += filterWidth / 2;
		indexY += filterWidth / 2;
		return indexY * newWidth + indexX;
	}



	template<typename T, typename int WIDTH>
	__global__ void convolutionGPU(processing::Filter<T, WIDTH> * filter, const int numRows, const int numCols, uchar * inputImage, T * outputImage, int maxFilterWidth)
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
		//if (index1D == (1628490))
		//{
#pragma unroll WIDTH
		for (int yOffset = 0; yOffset < WIDTH; yOffset++)
		{
#pragma unroll WIDTH
			for (int xOffset = 0; xOffset < WIDTH; xOffset++)
			{
				pointPosition.x = absoluteImagePosition.x + xOffset - WIDTH / 2;
				pointPosition.y = absoluteImagePosition.y + yOffset - WIDTH / 2;
				result += filterV[yOffset*WIDTH + xOffset] * inputImage[indexInNew(pointPosition.x, pointPosition.y, numCols, numRows, maxFilterWidth)];
				//printf("Result: %f\n", result);
			}
		}

		outputImage[index1D] = result;
		//}

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
	KernelSlowConvolutionNoEdgeCopy<T>::KernelSlowConvolutionNoEdgeCopy(vector<shared_ptr<AbstractFilter<T>>>& filters) :
		h_filters_(filters)
	{

	}





	template<typename T>
	void KernelSlowConvolutionNoEdgeCopy<T>::run(ImageFactory & image, vector<shared_ptr<T>>& results)
	{
	uint filterCount(h_filters_.size());
	size_t memmoryToAllocateForFiltersOnDevice(0);
	for_each(h_filters_.begin(), h_filters_.end(), [&memmoryToAllocateForFiltersOnDevice](auto& filter) { memmoryToAllocateForFiltersOnDevice += filter->getSize(); });
	shared_ptr<uchar> deviceFilters = allocateMemmoryDevice<uchar>( memmoryToAllocateForFiltersOnDevice);
	uint offset(0);
	for_each(h_filters_.begin(), h_filters_.end(), [&deviceFilters, &offset](auto& filter)
	{
	filter->copyWholeFilterToDeviceMemory(deviceFilters.get() + offset);
	offset += filter->getSize();
	});
	// filter allocation and initialization
	shared_ptr<uchar4> deviceRGBAImage = allocateMemmoryDevice<uchar4>(image.getNumPixels());

	shared_ptr<uchar> deviceRedChannelIn = allocateMemmoryDevice<uchar>(image.getNumPixels());
	shared_ptr<uchar> deviceGreenChannelIn = allocateMemmoryDevice<uchar>(image.getNumPixels());
	shared_ptr<uchar> deviceBlueChannelIn = allocateMemmoryDevice<uchar>(image.getNumPixels());

	shared_ptr<uchar> deviceRedChannelOut = allocateMemmoryDevice<uchar>(image.getNumPixels());
	shared_ptr<uchar> deviceGreenChannelOut = allocateMemmoryDevice<uchar>(image.getNumPixels());
	shared_ptr<uchar> deviceBlueChannelOut = allocateMemmoryDevice<uchar>(image.getNumPixels());

	uchar4* hostRGBAImage = image.getInputRGBAPointer();
	checkCudaErrors(cudaMemcpy(deviceRGBAImage.get(), hostRGBAImage, image.getNumPixels() * sizeof(uchar4), cudaMemcpyHostToDevice));
	// memory allocation
	const uint numberOfThreadsInBlock = 32;
	const dim3 blockSize(numberOfThreadsInBlock, numberOfThreadsInBlock);
	const dim3 gridSize((image.getNumCols() + blockSize.x - 1) / blockSize.x, (image.getNumRows() + blockSize.y - 1) / blockSize.y, 1);
	// kernels parameters
	separateChannels<char> << <gridSize, blockSize >> > (deviceRGBAImage.get(), image.getNumRows(), image.getNumCols(), deviceRedChannelIn.get(), deviceGreenChannelIn.get(), deviceBlueChannelIn.get());
	checkCudaErrors(cudaDeviceSynchronize());
	// initialization of channels
	offset = 0;
	for (auto& filter : h_filters_)
	{
	switch (filter->getWidth())
	{
	case 3:
	{
	Filter<T, 3> * ptr = (Filter<T, 3> *) (deviceFilters.get() + offset);
	convolution <<<gridSize, blockSize>>>(ptr, image.getNumRows(), image.getNumCols(), deviceRedChannelIn.get(), deviceRedChannelOut.get());
	checkCudaErrors(cudaDeviceSynchronize());
	convolution << <gridSize, blockSize >> >(ptr, image.getNumRows(), image.getNumCols(), deviceGreenChannelIn.get(), deviceGreenChannelOut.get());
	checkCudaErrors(cudaDeviceSynchronize());
	convolution << <gridSize, blockSize >> >(ptr, image.getNumRows(), image.getNumCols(), deviceBlueChannelIn.get(), deviceBlueChannelOut.get());
	checkCudaErrors(cudaDeviceSynchronize());
	break;
	}
	case 5:
	{
	Filter<T, 5> * ptr = (Filter<T, 5> *) (deviceFilters.get() + offset);
	convolution << <gridSize, blockSize >> >(ptr, image.getNumRows(), image.getNumCols(), deviceRedChannelIn.get(), deviceRedChannelOut.get());
	checkCudaErrors(cudaDeviceSynchronize());
	convolution << <gridSize, blockSize >> >(ptr, image.getNumRows(), image.getNumCols(), deviceGreenChannelIn.get(), deviceGreenChannelOut.get());
	checkCudaErrors(cudaDeviceSynchronize());
	convolution << <gridSize, blockSize >> >(ptr, image.getNumRows(), image.getNumCols(), deviceBlueChannelIn.get(), deviceBlueChannelOut.get());
	checkCudaErrors(cudaDeviceSynchronize());
	break;
	}
	case 7:
	{
	Filter<T, 7> * ptr = (Filter<T, 7> *) (deviceFilters.get() + offset);
	convolution << <gridSize, blockSize >> >(ptr, image.getNumRows(), image.getNumCols(), deviceRedChannelIn.get(), deviceRedChannelOut.get());
	checkCudaErrors(cudaDeviceSynchronize());
	convolution << <gridSize, blockSize >> >(ptr, image.getNumRows(), image.getNumCols(), deviceGreenChannelIn.get(), deviceGreenChannelOut.get());
	checkCudaErrors(cudaDeviceSynchronize());
	convolution << <gridSize, blockSize >> >(ptr, image.getNumRows(), image.getNumCols(), deviceBlueChannelIn.get(), deviceBlueChannelOut.get());
	checkCudaErrors(cudaDeviceSynchronize());
	break;
	}
	default:
	break;
	}
	offset += filter->getSize();
	recombineChannels<char><<<gridSize, blockSize>>>(deviceRedChannelOut.get(), deviceGreenChannelOut.get(), deviceBlueChannelOut.get(), deviceRGBAImage.get(), image.getNumRows(), image.getNumCols());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors( cudaMemcpy(image.getOutputRGBAPointer(), deviceRGBAImage.get(), image.getNumPixels() * sizeof(uchar4), cudaMemcpyDeviceToHost) );

	}
	}
	
}
