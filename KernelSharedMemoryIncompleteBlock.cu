#include "KernelSharedMemoryIncompleteBlock.h"

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
#include <type_traits>

using namespace std;

#define MAX_SMALL_TILE_DIMENION_X 3
#define MAX_SMALL_TILE_DIMENION_Y 3

#define CONVOLUTIONSHAREDINCOMPLETEBLOCK(FILTERWIDTH, BLOCKSIZEX, BLOCKSIZEY, TILESIZEX, TILESIZEY) \
			case FILTERWIDTH:\
			{\
				checkCudaErrors(cudaMemcpyToSymbol(FILTERCUDA, filter->getFilter(), sizeof(float) * FILTERWIDTH * FILTERWIDTH));\
				const int FILTER_WIDTH = FILTERWIDTH;\
				const int BLOCK_SIZE_X = BLOCKSIZEX;\
				const int BLOCK_SIZE_Y = BLOCKSIZEY;\
				const int TILE_SIZE_X = TILESIZEX;\
				const int TILE_SIZE_Y = TILESIZEY;\
				const dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y); \
				const dim3 gridSize((colsForGridX + TILE_SIZE_X - 1) / TILE_SIZE_X, (rowsForGridY + TILE_SIZE_Y - 1) / TILE_SIZE_Y, 1);  \
				static_assert(BLOCK_SIZE_X * MAX_SMALL_TILE_DIMENION_X - TILE_SIZE_X * MAX_SMALL_TILE_DIMENION_X >= FILTER_WIDTH - 1, "Wrong block and tile size: BLOCK_SIZE_X * MAX_SMALL_TILE_DIMENION_X - TILE_SIZE_X * MAX_SMALL_TILE_DIMENION_X >= FILTER_WIDTH - 1"); \
				static_assert(BLOCK_SIZE_Y * MAX_SMALL_TILE_DIMENION_Y - TILE_SIZE_Y * MAX_SMALL_TILE_DIMENION_Y >= FILTER_WIDTH - 1, "Wrong block and tile size: BLOCK_SIZE_Y * MAX_SMALL_TILE_DIMENION_Y - TILE_SIZE_Y * MAX_SMALL_TILE_DIMENION_Y >= FILTER_WIDTH - 1"); \
				convolutionGPUSharedIncompleteBlock< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y> << <gridSize, blockSize >> >(deviceGrayImageIn, deviceGrayImageOut, inputPitch / sizeof(float), outputPitch / sizeof(float)); \
				int xlen = image.getNumCols() - (FILTER_WIDTH - 1); \
				int ylen = image.getNumRows() - (FILTER_WIDTH - 1); \
				shared_ptr<float> resultCPU = makeArray<float>(xlen*ylen); \
				checkCudaErrors(cudaHostRegister(resultCPU.get(), xlen*ylen * sizeof(float), cudaHostRegisterPortable)); \
				checkCudaErrors(cudaMemcpy2D(resultCPU.get(), xlen * sizeof(float), deviceGrayImageOut, outputPitch, xlen * sizeof(float), ylen, cudaMemcpyDeviceToHost)); \
				checkCudaErrors(cudaHostUnregister(resultCPU.get())); \
				results.push_back(resultCPU); \
				break; \
			}

#define IMAD(a, b, c) ( __mul24((a), (b)) + (c) )
#define CEIL(a, b) ((a + b - 1) / b)


namespace processing
{
	
	
	template<typename int FILTER_WIDTH, typename int BLOCK_SIZE_X, typename int BLOCK_SIZE_Y, typename int TILE_SIZE_X, typename int TILE_SIZE_Y>
	__global__ void convolutionGPUSharedIncompleteBlock(float * __restrict__ inputImage, float * __restrict__  outputImage, int inputPitch, int outputPitch)
	{
		__shared__ float shared[BLOCK_SIZE_Y * MAX_SMALL_TILE_DIMENION_Y][BLOCK_SIZE_X * MAX_SMALL_TILE_DIMENION_X];
		int threadX = threadIdx.x * MAX_SMALL_TILE_DIMENION_X;
		int threadY = threadIdx.y * MAX_SMALL_TILE_DIMENION_Y;
		int2 absoluteImagePosition;
		absoluteImagePosition.x = IMAD(blockIdx.x, TILE_SIZE_X, threadIdx.x) * MAX_SMALL_TILE_DIMENION_X;
		absoluteImagePosition.y = IMAD(blockIdx.y, TILE_SIZE_Y, threadIdx.y) * MAX_SMALL_TILE_DIMENION_Y;
#pragma unroll MAX_SMALL_TILE_DIMENION_Y
		for (int i = 0; i < MAX_SMALL_TILE_DIMENION_Y; i++)
		{
#pragma unroll MAX_SMALL_TILE_DIMENION_X
			for (int j = 0; j < MAX_SMALL_TILE_DIMENION_X; j++)
			{
				shared[threadY + i][threadX + j] = inputImage[IMAD(absoluteImagePosition.y + i, inputPitch, absoluteImagePosition.x + j)];
			}
		}
		__syncthreads();
		if (threadX < TILE_SIZE_X * MAX_SMALL_TILE_DIMENION_X  && threadY < TILE_SIZE_Y * MAX_SMALL_TILE_DIMENION_Y)
		{
			float results[MAX_SMALL_TILE_DIMENION_X * MAX_SMALL_TILE_DIMENION_Y] = { 0.0 };
			float filterValue;
#pragma unroll FILTER_WIDTH
			for (int yOffset = 0; yOffset < FILTER_WIDTH; yOffset++)
			{
#pragma unroll FILTER_WIDTH
				for (int xOffset = 0; xOffset < FILTER_WIDTH; xOffset++)
				{
					filterValue = FILTERCUDA[yOffset*FILTER_WIDTH + xOffset];
#pragma unroll MAX_SMALL_TILE_DIMENION_Y
					for (int i = 0; i < MAX_SMALL_TILE_DIMENION_Y; i++)
					{
#pragma unroll MAX_SMALL_TILE_DIMENION_X
						for (int j = 0; j < MAX_SMALL_TILE_DIMENION_X; j++)
						{
							results[i *MAX_SMALL_TILE_DIMENION_Y + j] += filterValue * shared[yOffset + threadY + i][xOffset + threadX + j];
						}

					}
				}
			}
#pragma unroll MAX_SMALL_TILE_DIMENION_Y
			for (int i = 0; i < MAX_SMALL_TILE_DIMENION_Y; i++)
			{
#pragma unroll MAX_SMALL_TILE_DIMENION_X
				for (int j = 0; j < MAX_SMALL_TILE_DIMENION_X; j++)
				{
					outputImage[IMAD(absoluteImagePosition.y + i, outputPitch, absoluteImagePosition.x + j)] = results[i *MAX_SMALL_TILE_DIMENION_Y + j];
				}
			}

		}
	}


	KernelSharedMemoryIncompleteBlock::KernelSharedMemoryIncompleteBlock()
	{}


	void KernelSharedMemoryIncompleteBlock::run(ImageFactory& image, vector<shared_ptr<Filter>>& filters, vector<shared_ptr<float>>& results)
	{
		const short MAX_BLOCK_SIZE_X = 64;
		const short MAX_BLOCK_SIZE_Y = 64;
		

		float* deviceGrayImageOut = 0;
		size_t outputPitch = 0;
		checkCudaErrors(cudaMallocPitch<float>(&deviceGrayImageOut, &outputPitch, (image.getNumCols() + MAX_BLOCK_SIZE_X * MAX_SMALL_TILE_DIMENION_X) * sizeof(float), (image.getNumRows() + MAX_BLOCK_SIZE_Y * MAX_SMALL_TILE_DIMENION_Y)));
		float* hostGrayImage = image.getInputGrayPointerFloat();

		// memory allocation
		int colsForGridX = CEIL(image.getNumCols(), MAX_SMALL_TILE_DIMENION_X);
		int rowsForGridY = CEIL(image.getNumRows(), MAX_SMALL_TILE_DIMENION_Y);

		float* deviceGrayImageIn;
		size_t inputPitch = 0;

		checkCudaErrors(cudaMallocPitch<float>(&deviceGrayImageIn, &inputPitch, (image.getNumCols() + MAX_BLOCK_SIZE_X * MAX_SMALL_TILE_DIMENION_X) * sizeof(float), image.getNumRows() + MAX_BLOCK_SIZE_Y* MAX_SMALL_TILE_DIMENION_Y));
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaMemcpy2D(deviceGrayImageIn, inputPitch, hostGrayImage, image.getNumCols() * sizeof(float), image.getNumCols() * sizeof(float), image.getNumRows(), cudaMemcpyHostToDevice));


		for (auto& filter : filters)
		{
			switch (filter->getWidth())
			{
				//CONVOLUTIONSHAREDSMALL(FILTERWIDTH, BLOCKSIZEX, BLOCKSIZEY, TILESIZEX, TILESIZEY)
				//CONVOLUTIONSHAREDINCOMPLETEBLOCK(9, 32, 32, 24, 24)
				//CONVOLUTIONSHAREDINCOMPLETEBLOCK(11, 32, 32, 22, 22)
				//CONVOLUTIONSHAREDINCOMPLETEBLOCK(13, 32, 32, 20, 20)
				//CONVOLUTIONSHAREDINCOMPLETEBLOCK(15, 32, 32, 18, 18)
				
				//CONVOLUTIONSHAREDINCOMPLETEBLOCK(1, 32, 16, 32, 16)
				//CONVOLUTIONSHAREDINCOMPLETEBLOCK(3, 32, 16, 30, 14)
			//	CONVOLUTIONSHAREDINCOMPLETEBLOCK(5, 32, 16, 28, 12)
				//CONVOLUTIONSHAREDINCOMPLETEBLOCK(7, 32, 32, 26, 26)
				//CONVOLUTIONSHAREDINCOMPLETEBLOCK(9, 32, 16, 29, 13)
				CONVOLUTIONSHAREDINCOMPLETEBLOCK(11, 32, 20, 28, 16)
				//CONVOLUTIONSHAREDINCOMPLETEBLOCK(13, 32, 20, 28, 16)
				//CONVOLUTIONSHAREDINCOMPLETEBLOCK(15, 32, 13, 27, 8)
				
				/* 2 x 2
				CONVOLUTIONSHAREDINCOMPLETEBLOCK(1, 32, 16, 32, 16)
					CONVOLUTIONSHAREDINCOMPLETEBLOCK(3, 32, 16, 31, 15)
					CONVOLUTIONSHAREDINCOMPLETEBLOCK(5, 32, 16, 30, 14)
					CONVOLUTIONSHAREDINCOMPLETEBLOCK(7, 32, 32, 29, 29)
				CONVOLUTIONSHAREDINCOMPLETEBLOCK(9, 32, 32, 28, 28)
				CONVOLUTIONSHAREDINCOMPLETEBLOCK(11, 32, 32, 27, 27)
				CONVOLUTIONSHAREDINCOMPLETEBLOCK(13, 32, 32, 26, 26)
				CONVOLUTIONSHAREDINCOMPLETEBLOCK(15, 32, 18, 25, 11)
					*/
			default:
				std::cerr << "Filter with width: " << filter->getWidth() << " not supported!" << endl;
				break;
			}
		}
		cudaFree(deviceGrayImageIn);
		cudaFree(deviceGrayImageOut);
	}

}

