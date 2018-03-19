#include "KernelNaive.h"

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

#define CONVOLUTIONSLOWNAIVE(FILTERWIDTH)\
case FILTERWIDTH:\
{\
	float * ptr =  (deviceFilters.get() + offset);\
	convolutionGPUNaive << <gridSize, blockSize >> >(ptr, image.getNumRows(), image.getNumCols(), deviceGrayImageIn.get(), deviceGrayImageOut.get(), FILTERWIDTH);\
	break;\
}

	
	__global__ void convolutionGPUNaive(float * filter, const int numRows, const int numCols, float * inputImage, float * outputImage, int filterWidth)
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
		for (int yOffset = 0; yOffset < filterWidth; yOffset++)
		{
			for (int xOffset = 0; xOffset < filterWidth; xOffset++)
			{
				pointPosition.x = absoluteImagePosition.x + xOffset - filterWidth / 2;
				pointPosition.y = absoluteImagePosition.y + yOffset - filterWidth / 2;
				pointPosition.x = KernelNaive::min(KernelNaive::max(pointPosition.x, 0), numCols - 1);
				pointPosition.y = KernelNaive::min(KernelNaive::max(pointPosition.y, 0), numRows - 1);
				result += filter[yOffset*filterWidth + xOffset] * inputImage[pointPosition.y*numCols + pointPosition.x];
			}
		}
		outputImage[index1D] = result;
	}


	KernelNaive::KernelNaive()
	{
	}


	void KernelNaive::run(ImageFactory& image, vector<shared_ptr<AbstractFilter>>& filters, vector<shared_ptr<float>>& results)
	{
		shared_ptr<float> deviceFilters = makeDeviceFilters(filters);
		
		// filter allocation and initialization
		shared_ptr<float> deviceGrayImageOut = allocateMemmoryDevice<float>(image.getNumPixels());
		const float * hostGrayImage = image.getInputGrayPointerFloat();

		shared_ptr<float> deviceGrayImageIn = allocateMemmoryDevice<float>(image.getNumPixels());
		checkCudaErrors(cudaMemcpy(deviceGrayImageIn.get(), hostGrayImage, image.getNumPixels() * sizeof(float), cudaMemcpyHostToDevice));
		// memory allocation

		const uint numberOfThreadsInBlock = 16;
		const dim3 blockSize(numberOfThreadsInBlock, numberOfThreadsInBlock);
		const dim3 gridSize((image.getNumCols() + blockSize.x - 1) / blockSize.x, (image.getNumRows() + blockSize.y - 1) / blockSize.y, 1);
		// kernels parameters
		uint offset(0);
		for (auto& filter : filters)
		{
			switch (filter->getWidth())
			{
			case 1:
			{
				float * ptr = (deviceFilters.get() + offset);

				convolutionGPUNaive << <gridSize, blockSize >> >(ptr, image.getNumRows(), image.getNumCols(), deviceGrayImageIn.get(), deviceGrayImageOut.get(), 1); 
				break; 
			}
				//CONVOLUTIONSLOWNAIVE(1)
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
			shared_ptr<float> resultCPU = makeArray<float>(image.getNumPixels());
			checkCudaErrors(cudaMemcpy(resultCPU.get(), deviceGrayImageOut.get(), image.getNumPixels() * sizeof(float), cudaMemcpyDeviceToHost));
			results.push_back(resultCPU);
		}
		checkCudaErrors(cudaDeviceSynchronize());
	}

}
