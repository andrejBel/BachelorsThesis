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
#include <algorithm>

using namespace std;
using namespace cv;

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
	__host__ __device__ __forceinline__  const T min(const T a, const T b) {
		return !(b<a) ? a : b;
	}

	template <typename T>
	__host__ __device__ __forceinline__ const T max(const T a, const T b) {
		return (b<a) ? a : b;
	}

	__device__ __forceinline__ int indexInNew(int indexX, int indexY, int originalWidth, int originalHeight, int filterWidth)
	{
		int newWidth = originalWidth + (filterWidth / 2) * 2;
		indexX += filterWidth / 2;
		indexY += filterWidth / 2;
		return indexY * newWidth + indexX;
	}

	template<typename T, typename int FILTER_WIDTH>
	__global__ void convolutionGPUSlow(processing::Filter<T, FILTER_WIDTH> * filter, const int numRows, const int numCols, uchar * inputImage, T * outputImage, int maxFilterWidth)
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
		#pragma unroll FILTER_WIDTH
		for (int yOffset = 0; yOffset < FILTER_WIDTH; yOffset++)
		{
		#pragma unroll FILTER_WIDTH
			for (int xOffset = 0; xOffset < FILTER_WIDTH; xOffset++)
			{
				pointPosition.x = absoluteImagePosition.x + xOffset - FILTER_WIDTH / 2;
				pointPosition.y = absoluteImagePosition.y + yOffset - FILTER_WIDTH / 2;
				result += filterV[yOffset*FILTER_WIDTH + xOffset] * inputImage[indexInNew(pointPosition.x, pointPosition.y, numCols, numRows, maxFilterWidth)];
				//printf("Result: %f\n", result);
			}
		}

		outputImage[index1D] = result;
		//}

	}


	template<typename T>
	__host__ __forceinline__ int KernelSlowConvolution<T>::indexToCopyToMirrored(int index, int numCols, int numRows, const int filterWidth)
	{
		int indexX = (index % (numCols + (filterWidth / 2) * 2)) - (filterWidth / 2);
		int indexY = (index / (numCols + (filterWidth / 2) * 2)) - (filterWidth / 2);
		indexX = processing::min(processing::max(indexX, 0), numCols - 1);
		indexY = processing::min(processing::max(indexY, 0), numRows - 1);
		return indexY * numCols + indexX;
	}



	template<typename T>
	KernelSlowConvolution<T>::KernelSlowConvolution()
	{}



	template<typename T>
	void KernelSlowConvolution<T>::run(ImageFactory& image, vector<shared_ptr<AbstractFilter<T>>>& filters, vector<shared_ptr<T>>& results)
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
		auto originalNumCols = image.getNumCols();
		auto originalNumRows = image.getNumRows();
		auto newNumCols = originalNumCols + (maxFilterWidth / 2) * 2;
		auto newNumRows = originalNumRows + (maxFilterWidth / 2) * 2;
		shared_ptr<uchar> deviceGrayImageIn = allocateMemmoryDevice<uchar>(newNumCols*newNumRows);
		shared_ptr<uchar> modifiedHostGrayImage = makeArray<uchar>(newNumCols*newNumRows);
		auto modifiedPtr = modifiedHostGrayImage.get();
		for (size_t i = 0; i < newNumCols*newNumRows; i++)
		{
			modifiedPtr[i] = hostGrayImage[indexToCopyToMirrored(i, originalNumCols, originalNumRows, maxFilterWidth)];
		}
		checkCudaErrors(cudaMemcpy(deviceGrayImageIn.get(), modifiedPtr, newNumCols*newNumRows * sizeof(uchar), cudaMemcpyHostToDevice));
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
			case 3:
			{
				Filter<T, 3> * ptr = (Filter<T, 3> *) (deviceFilters.get() + offset);
				convolutionGPUSlow << <gridSize, blockSize >> >(ptr, image.getNumRows(), image.getNumCols(), deviceGrayImageIn.get(), deviceGrayImageOut.get(), maxFilterWidth);
				checkCudaErrors(cudaDeviceSynchronize());
				break;
			}
			case 5:
			{
				Filter<T, 5> * ptr = (Filter<T, 5> *) (deviceFilters.get() + offset);
				convolutionGPUSlow << <gridSize, blockSize >> >(ptr, image.getNumRows(), image.getNumCols(), deviceGrayImageIn.get(), deviceGrayImageOut.get(), maxFilterWidth);
				checkCudaErrors(cudaDeviceSynchronize());

				break;
			}
			case 7:
			{
				Filter<T, 7> * ptr = (Filter<T, 7> *) (deviceFilters.get() + offset);
				convolutionGPUSlow << <gridSize, blockSize >> >(ptr, image.getNumRows(), image.getNumCols(), deviceGrayImageIn.get(), deviceGrayImageOut.get(), maxFilterWidth);
				checkCudaErrors(cudaDeviceSynchronize());
				break;
			}
			default:
				break;
			}
			offset += filter->getSize();
			shared_ptr<T> resultCPU = makeArray<T>(image.getNumPixels());
			checkCudaErrors(cudaMemcpy(resultCPU.get(), deviceGrayImageOut.get(), image.getNumPixels() * sizeof(T), cudaMemcpyDeviceToHost));
			results.push_back(resultCPU);
			//image.copyDeviceGrayToHostGrayOut(deviceGrayImageOut.get());
			//image.saveGrayImgOut("blurredImage.jpg");
		}
		cout << "";
	}



	/* color
	template<typename T>
	void KernelSlowConvolution<T>::run(ImageFactory & image, vector<shared_ptr<T>>& results)
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


	image.saveRGBAImgOut("blurredImage.jpg");
	}
	}
	*/
}
