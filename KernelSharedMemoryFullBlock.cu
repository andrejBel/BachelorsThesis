#include "KernelSharedMemoryFullBlock.h"

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

#include "GpuTimer.h"


using namespace std;
/*
CONVOLUTIONSHARED(1, 32, 4)
CONVOLUTIONSHARED(3, 32, 4)
CONVOLUTIONSHARED(5, 32, 4)
CONVOLUTIONSHARED(7, 32, 4) // najlepsie CONVOLUTIONSHARED(7, 32, 4) 2x2
CONVOLUTIONSHARED(9, 32, 4) // najlepsie CONVOLUTIONSHARED(9, 32, 4) 2x2
CONVOLUTIONSHARED(11, 32, 8) // najlepsie CONVOLUTIONSHARED(11, 32, 8) 2x2
CONVOLUTIONSHARED(13, 32, 10) // najlepsie CONVOLUTIONSHARED(11, 32, 8) 2x2
CONVOLUTIONSHARED(15, 32, 10) // najlepsie CONVOLUTIONSHARED(11, 32, 8) 2x2

CONVOLUTIONSHARED(13, 32, 6) // 3 x 3 jedine pouzitelne
CONVOLUTIONSHARED(15, 32, 8) // 3 x 3 jedine pouzitelne
*/

#define TILE_SIZE_X 3
#define TILE_SIZE_Y 3

#define CONVOLUTIONSHARED(FILTER_W, BLOCK_X, BLOCK_Y)\
case FILTER_W:\
{\
	checkCudaErrors(cudaMemcpyToSymbol(FILTERCUDA, filter->getFilter(), sizeof(float) * FILTER_W * FILTER_W));\
	const int BLOCK_SIZE_X = BLOCK_X;\
	const int BLOCK_SIZE_Y = BLOCK_Y;\
	const int FILTER_WIDTH = FILTER_W;\
	const dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y);\
	const dim3 gridSize((image.getNumCols() / TILE_SIZE_X + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (image.getNumRows() / TILE_SIZE_Y + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y, 1);\
	convolutionGPUSharedFullBlock<FILTER_WIDTH, BLOCK_SIZE_X,BLOCK_SIZE_Y> << <gridSize, blockSize >> >(deviceGrayImageIn, deviceGrayImageOut, inputPitch / sizeof(float), outputPitch / sizeof(float));\
	int xlen = image.getNumCols() - (FILTER_WIDTH - 1); \
	int ylen = image.getNumRows() - (FILTER_WIDTH - 1); \
	shared_ptr<float> resultCPU = makeArray<float>(xlen*ylen); \
	checkCudaErrors(cudaHostRegister(resultCPU.get(), xlen*ylen * sizeof(float), cudaHostRegisterPortable)); \
	checkCudaErrors(cudaMemcpy2D(resultCPU.get(), xlen * sizeof(float), deviceGrayImageOut, outputPitch, xlen * sizeof(float), ylen, cudaMemcpyDeviceToHost)); \
	checkCudaErrors(cudaHostUnregister(resultCPU.get())); \
	results.push_back(resultCPU); \
	break;\
}

#define IMAD(a, b, c) ( __mul24((a), (b)) + (c) )

namespace processing
{
	template<typename int FILTER_WIDTH, typename int BLOCK_SIZE_X, typename int BLOCK_SIZE_Y>
	__global__ void convolutionGPUSharedFullBlock(float * inputImage, float * outputImage, const size_t inputPitch, const size_t outputPitch)
	{
		int2 absoluteImagePosition;
		absoluteImagePosition.x = IMAD(blockIdx.x, BLOCK_SIZE_X, threadIdx.x) * TILE_SIZE_X;
		absoluteImagePosition.y = IMAD(blockIdx.y, BLOCK_SIZE_Y, threadIdx.y) * TILE_SIZE_Y;
		int2 positionShared;
		positionShared.x = blockIdx.x * BLOCK_SIZE_X * TILE_SIZE_X;
		positionShared.y = blockIdx.y * BLOCK_SIZE_Y * TILE_SIZE_Y;
		__shared__ float shared[BLOCK_SIZE_Y*TILE_SIZE_Y + (FILTER_WIDTH - 1)
#if TILE_SIZE_Y > 2
	+ TILE_SIZE_Y
#endif
		][BLOCK_SIZE_X *TILE_SIZE_X + (FILTER_WIDTH - 1)
#if TILE_SIZE_X > 2
	+ TILE_SIZE_X
#endif
		];
		int threadX = threadIdx.x * TILE_SIZE_X;
		int threadY = threadIdx.y * TILE_SIZE_Y;
		for (int j = threadY; j < FILTER_WIDTH - 1 + BLOCK_SIZE_Y * TILE_SIZE_Y; j += BLOCK_SIZE_Y * TILE_SIZE_Y)
		{
			for (int i = threadX; i < FILTER_WIDTH - 1 + BLOCK_SIZE_X * TILE_SIZE_X; i+= BLOCK_SIZE_X * TILE_SIZE_X)
			{
#pragma unroll TILE_SIZE_Y
				for (int k = 0; k < TILE_SIZE_Y; k++)
				{
#pragma unroll TILE_SIZE_X
					for (int l = 0; l < TILE_SIZE_X; l++)
					{
						shared[j + k][i + l] = inputImage[IMAD(positionShared.y + j + k, inputPitch, positionShared.x + i + l)];
					}
				}
				
			}
		}
		__syncthreads();
		float results[TILE_SIZE_X * TILE_SIZE_Y] = {0.0};
		float filterValue;
#pragma unroll FILTER_WIDTH
		for (int yOffset = 0; yOffset < FILTER_WIDTH; yOffset++)
		{
#pragma unroll FILTER_WIDTH
			for (int xOffset = 0; xOffset < FILTER_WIDTH; xOffset++)
			{
				filterValue = FILTERCUDA[yOffset * FILTER_WIDTH + xOffset];
#pragma unroll TILE_SIZE_Y
				for (int k = 0; k < TILE_SIZE_Y; k++)
				{
#pragma unroll TILE_SIZE_X
					for (int l = 0; l < TILE_SIZE_X; l++)
					{
						results[k * TILE_SIZE_Y + l] += filterValue * shared[yOffset + threadY + k][xOffset + threadX + l];
					}
				}
			}
		}
#pragma unroll TILE_SIZE_Y
		for (int k = 0; k < TILE_SIZE_Y; k++)
		{
#pragma unroll TILE_SIZE_X
			for (int l = 0; l < TILE_SIZE_X; l++)
			{
				outputImage[IMAD(absoluteImagePosition.y + k, outputPitch, absoluteImagePosition.x + l)] = results[k * TILE_SIZE_Y + l];
			}
		}
		
	}

	
	KernelSharedMemoryFullBlock::KernelSharedMemoryFullBlock() 
	{}

	
	void KernelSharedMemoryFullBlock::run(ImageFactory& image, vector<shared_ptr<Filter>>& filters, vector<shared_ptr<float>>& results)
	{
		
		const short MAX_BLOCK_SIZE_X = 32;
		const short MAX_BLOCK_SIZE_Y = 32;

		const float* hostGrayImage = image.getInputGrayPointerFloat();

		float* deviceGrayImageOut = 0;
		size_t outputPitch = 0;
		checkCudaErrors(cudaMallocPitch<float>(&deviceGrayImageOut, &outputPitch, (image.getNumCols() + MAX_BLOCK_SIZE_X * TILE_SIZE_X * 10) * sizeof(float), (image.getNumRows() + MAX_BLOCK_SIZE_Y * TILE_SIZE_Y * 10)));
		// memory allocation
		size_t inputPitch = 0;
		float* deviceGrayImageIn;
		checkCudaErrors(cudaMallocPitch<float>(&deviceGrayImageIn, &inputPitch, (image.getNumCols() + MAX_BLOCK_SIZE_X * TILE_SIZE_X * 10) * sizeof(float), image.getNumRows() + MAX_BLOCK_SIZE_Y* TILE_SIZE_Y * 10));
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaMemcpy2D(deviceGrayImageIn, inputPitch, hostGrayImage, image.getNumCols() * sizeof(float), image.getNumCols() * sizeof(float), image.getNumRows(), cudaMemcpyHostToDevice));


		for (auto& filter : filters)
		{
			switch (filter->getWidth())
			{
			
			//CONVOLUTIONSHARED(1, 32,4)
			//CONVOLUTIONSHARED(3, 32, 4)
			//CONVOLUTIONSHARED(5, 32,4) 
			//CONVOLUTIONSHARED(7, 32, 4) // najlepsie CONVOLUTIONSHARED(7, 32, 4) 2x2
			
				CONVOLUTIONSHARED(11, 32, 7)
			CONVOLUTIONSHARED(13, 32, 6) 
			CONVOLUTIONSHARED(15, 32, 8) 
			
				/*
				CONVOLUTIONSHARED(1, 32, 16)
				CONVOLUTIONSHARED(3, 32, 16)
				CONVOLUTIONSHARED(5, 32, 16)
				CONVOLUTIONSHARED(7, 32, 16)
				CONVOLUTIONSHARED(9, 32, 16)
				CONVOLUTIONSHARED(11, 32, 16)
				CONVOLUTIONSHARED(13, 32, 16)
				CONVOLUTIONSHARED(15, 32, 16)
				*/
			default:
				std::cerr << "Filter with width: " << filter->getWidth() << " not supported!" << endl;
				break;
			}
		}
		checkCudaErrors(cudaFree(deviceGrayImageIn));
		checkCudaErrors(cudaFree(deviceGrayImageOut));
	}

}

