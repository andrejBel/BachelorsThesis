#include "KernelNaiveImproved.h"

#include <cuda.h>
#include <helper_cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

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


#define CONVOLUTIONNAIVEIMPROVED(FILTERWIDTH)\
case FILTERWIDTH:\
{\
	checkCudaErrors(cudaMemcpyToSymbol(FILTERCUDA, filter->getFilter(), sizeof(float) * FILTERWIDTH * FILTERWIDTH));\
	convolutionGPU<FILTERWIDTH> << <gridSize, blockSize >> >(image.getNumRows(), image.getNumCols(), deviceGrayImageIn.get(), deviceGrayImageOut.get());\
	break;\
}

	template<typename int FILTER_WIDTH>
	__global__ void convolutionGPU(const int numRows, const int numCols, float * inputImage, float * outputImage)
	{
		int2 absoluteImagePosition;

		absoluteImagePosition.x = blockIdx.x * blockDim.x + threadIdx.x;
		absoluteImagePosition.y = blockIdx.y * blockDim.y + threadIdx.y;
		if (absoluteImagePosition.x >= numCols || absoluteImagePosition.y >= numRows)
		{
			return;
		}
		const size_t index1D = absoluteImagePosition.y * numCols + absoluteImagePosition.x;
		float result(0.0);
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
				result += FILTERCUDA[yOffset*FILTER_WIDTH + xOffset] * inputImage[pointPosition.y*numCols + pointPosition.x];
			}
		}
		outputImage[index1D] = result;
	}


	__host__ __forceinline__ int KernelNaiveImproved::indexToCopyToMirrored(int index, int numCols, int numRows, const int filterWidth)
	{
		int indexX = (index % (numCols + (filterWidth / 2) * 2)) - (filterWidth / 2);
		int indexY = (index / (numCols + (filterWidth / 2) * 2)) - (filterWidth / 2);
		indexX = std::min(std::max(indexX, 0), numCols - 1);
		indexY = std::min(std::max(indexY, 0), numRows - 1);
		return indexY * numCols + indexX;
	}



	KernelNaiveImproved::KernelNaiveImproved() 
	{
	}


	void KernelNaiveImproved::run(ImageFactory& image, vector<shared_ptr<Filter>>& filters, vector<shared_ptr<float>>& results)
	{
		// filter allocation and initialization
		shared_ptr<float> deviceGrayImageOut = allocateMemmoryDevice<float>(image.getNumPixels());
		const float* hostGrayImage = image.getInputGrayPointerFloat();

		shared_ptr<float> deviceGrayImageIn = allocateMemmoryDevice<float>(image.getNumPixels());
		checkCudaErrors(cudaMemcpy(deviceGrayImageIn.get(), hostGrayImage, image.getNumPixels() * sizeof(float), cudaMemcpyHostToDevice));
		// memory allocation

		const uint numberOfThreadsInBlock = 16;
		const dim3 blockSize(numberOfThreadsInBlock, numberOfThreadsInBlock);
		const dim3 gridSize((image.getNumCols() + blockSize.x - 1) / blockSize.x, (image.getNumRows() + blockSize.y - 1) / blockSize.y, 1);
		// kernels parameters
		for (auto& filter : filters)
		{
			switch (filter->getWidth())
			{
				CONVOLUTIONNAIVEIMPROVED(1)
				CONVOLUTIONNAIVEIMPROVED(3)
				CONVOLUTIONNAIVEIMPROVED(5)
				CONVOLUTIONNAIVEIMPROVED(7)
				CONVOLUTIONNAIVEIMPROVED(9)
				CONVOLUTIONNAIVEIMPROVED(11)
				CONVOLUTIONNAIVEIMPROVED(13)
				CONVOLUTIONNAIVEIMPROVED(15)
			default:
				std::cerr << "Filter with width: " << filter->getWidth() << " not supported!" << endl;
				break;
			}
			shared_ptr<float> resultCPU = makeArray<float>(image.getNumPixels());
			checkCudaErrors(cudaDeviceSynchronize());
			checkCudaErrors(cudaMemcpy(resultCPU.get(), deviceGrayImageOut.get(), image.getNumPixels() * sizeof(float), cudaMemcpyDeviceToHost));
			results.push_back(resultCPU);		
		}
	}
	
}
