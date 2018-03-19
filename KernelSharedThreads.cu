#include "KernelSharedThreads.h"

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

#include "MemoryPoolPinned.h"
#include "MemoryPoolPitched.h"

using namespace std;


#define IMAD(a, b, c) ( __mul24((a), (b)) + (c) )
#define CEIL(a, b) ((a + b - 1) / b)
#define ROUNDUP(a, b) (((a + b - 1) / b) * b)


#define CONVOLUTIONSHAREDSMALL(FILTERWIDTH, BLOCKSIZEX, BLOCKSIZEY, TILESIZEX, TILESIZEY) \
			case FILTERWIDTH:\
			{\
				checkCudaErrors(cudaMemcpyToSymbol(FILTERCUDA, ((Filter<FILTERWIDTH> *) filter.get())->getFilter(), sizeof(float) * FILTERWIDTH * FILTERWIDTH));\
				const int FILTER_WIDTH = FILTERWIDTH;\
				const int BLOCK_SIZE_X = BLOCKSIZEX;\
				const int BLOCK_SIZE_Y = BLOCKSIZEY;\
				const int TILE_SIZE_X = TILESIZEX;\
				const int TILE_SIZE_Y = TILESIZEY;\
const dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y); \
const dim3 gridSize((colsForGridX + TILE_SIZE_X - 1) / TILE_SIZE_X, (rowsForGridY + TILE_SIZE_Y - 1) / TILE_SIZE_Y, 1);  \
convolutionGPUSharedThreads< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y> << <gridSize, blockSize >> >(image.getNumRows(), image.getNumCols(), deviceGrayImageIn, deviceGrayImageOut, inputPitch / sizeof(float), outputPitch / sizeof(float)); \
int xlen = image.getNumCols() - (FILTER_WIDTH - 1); \
int ylen = image.getNumRows() - (FILTER_WIDTH - 1); \
shared_ptr<float> resultCPU = MemoryPoolPinned::getMemoryPoolPinnedForOutput().acquireMemory(); \
checkCudaErrors(cudaMemcpy2D(resultCPU.get(), xlen * sizeof(float), deviceGrayImageOut, outputPitch, xlen * sizeof(float), ylen, cudaMemcpyDeviceToHost)); \
results.push_back(resultCPU); \
break; \
}

namespace processing
{


	template<typename int FILTER_WIDTH, typename int BLOCK_SIZE_X, typename int BLOCK_SIZE_Y, typename int TILE_SIZE_X, typename int TILE_SIZE_Y>
	__global__ void convolutionGPUSharedThreads(const int numRows, const int numCols, const float * __restrict__ inputImage, float * __restrict__  outputImage, size_t inputPitch, size_t outputPitch)
	{
		const int smallTile = 2;
		__shared__ float shared[BLOCK_SIZE_Y * smallTile][BLOCK_SIZE_X * smallTile];
		const int threadX = threadIdx.x * smallTile;
		const int threadY = threadIdx.y * smallTile;
		int2 absoluteImagePosition;
		absoluteImagePosition.x = IMAD(blockIdx.x, TILE_SIZE_X, threadIdx.x) * smallTile;
		absoluteImagePosition.y = IMAD(blockIdx.y, TILE_SIZE_Y, threadIdx.y) * smallTile;
		float2 firstRow = *(float2 *)(inputImage + IMAD(absoluteImagePosition.y, inputPitch, absoluteImagePosition.x));
		float2 secondRow = *(float2 *)(inputImage + IMAD(absoluteImagePosition.y + 1, inputPitch, absoluteImagePosition.x));
		shared[threadY][threadX] = firstRow.x;
		shared[threadY][threadX + 1] = firstRow.y;
		shared[threadY + 1][threadX] = secondRow.x;
		shared[threadY + 1][threadX + 1] = secondRow.y;
		__syncthreads();

		if (threadX < TILE_SIZE_X * smallTile && threadY < TILE_SIZE_Y * smallTile)
		{
			float result1 = 0.0; //00
			float result2 = 0.0; //01
			float result3 = 0.0; //10
			float result4 = 0.0; //11
			float filterValue = 0.0;
#pragma unroll FILTER_WIDTH
			for (int yOffset = 0; yOffset < FILTER_WIDTH; yOffset++)
			{
#pragma unroll FILTER_WIDTH
				for (int xOffset = 0; xOffset < FILTER_WIDTH; xOffset++)
				{
					filterValue = FILTERCUDA[yOffset*FILTER_WIDTH + xOffset];
					result1 += filterValue * shared[yOffset + threadY][xOffset + threadX];
					result2 += filterValue * shared[yOffset + threadY][xOffset + threadX + 1];
					result3 += filterValue * shared[yOffset + threadY + 1][xOffset + threadX];
					result4 += filterValue * shared[yOffset + threadY + 1][xOffset + threadX + 1];
				}
			}
			outputImage[IMAD(absoluteImagePosition.y, outputPitch, absoluteImagePosition.x)] = result1;
			outputImage[IMAD(absoluteImagePosition.y, outputPitch, absoluteImagePosition.x + 1)] = result2;
			outputImage[IMAD(absoluteImagePosition.y + 1, outputPitch, absoluteImagePosition.x)] = result3;
			outputImage[IMAD(absoluteImagePosition.y + 1, outputPitch, absoluteImagePosition.x + 1)] = result4;
		}
	}



	KernelSharedThreads::KernelSharedThreads()
	{}


	void KernelSharedThreads::run(ImageFactory& image, vector<shared_ptr<AbstractFilter>>& filters, vector<shared_ptr<float>>& results)
	{
		const short MAX_BLOCK_SIZE_X = 64;
		const short MAX_BLOCK_SIZE_Y = 64;
		const short MAX_TILE_SIZE_X = 62;
		const short MAX_TILE_SIZE_Y = 62;
		const short MAX_SMALL_TILE_DIMENION_X = 2;
		const short MAX_SMALL_TILE_DIMENION_Y = 2;
		float* deviceGrayImageOut = MemoryPoolPitched::getMemoryPoolPitchedForOutput().getMemory().at(0);
		size_t outputPitch = MemoryPoolPitched::getMemoryPoolPitchedForOutput().getPitch();
		
		float* hostGrayImage = image.getInputGrayPointerFloat();

		// memory allocation
		int colsForGridX = CEIL(image.getNumCols(), MAX_SMALL_TILE_DIMENION_X);
		int rowsForGridY = CEIL(image.getNumRows(), MAX_SMALL_TILE_DIMENION_Y);

		float* deviceGrayImageIn;
		size_t inputPitch = MemoryPoolPitched::getMemoryPoolPitchedForInput().getPitch();

		deviceGrayImageIn = MemoryPoolPitched::getMemoryPoolPitchedForInput().getMemory().at(0);
		checkCudaErrors(cudaMemcpy2D(deviceGrayImageIn, inputPitch, hostGrayImage, image.getNumCols() * sizeof(float), image.getNumCols() * sizeof(float), image.getNumRows(), cudaMemcpyHostToDevice));


		for (auto& filter : filters)
		{
			switch (filter->getWidth())
			{
				//CONVOLUTIONSHAREDSMALL(FILTERWIDTH, BLOCKSIZEX, BLOCKSIZEY, TILESIZEX, TILESIZEY)

				CONVOLUTIONSHAREDSMALL(1, 32, 16, 32, 16)
					CONVOLUTIONSHAREDSMALL(3, 32, 16, 31, 15)
					CONVOLUTIONSHAREDSMALL(5, 32, 16, 30, 14)
					CONVOLUTIONSHAREDSMALL(7, 32, 32, 29, 29)
					CONVOLUTIONSHAREDSMALL(9, 32, 32, 28, 28)
					CONVOLUTIONSHAREDSMALL(11, 32, 32, 27, 27)
					CONVOLUTIONSHAREDSMALL(13, 32, 32, 26, 26)
					CONVOLUTIONSHAREDSMALL(15, 32, 16, 25, 9)


			default:
				std::cerr << "Filter with width: " << filter->getWidth() << " not supported!" << endl;
				break;
			}
		}
	}

}


