#include "KernelSharedForSmall.h"

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
convolutionGPUSharedSmall< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y> << <gridSize, blockSize >> >(image.getNumRows(), image.getNumCols(), deviceGrayImageIn, deviceGrayImageOut, inputPitch / sizeof(float), outputPitch / sizeof(float)); \
int xlen = image.getNumCols() - (FILTER_WIDTH - 1); \
int ylen = image.getNumRows() - (FILTER_WIDTH - 1); \
shared_ptr<float> resultCPU = makeArray<float>(xlen*ylen); \
checkCudaErrors(cudaHostRegister(resultCPU.get(), xlen*ylen * sizeof(float), cudaHostRegisterPortable)); \
checkCudaErrors(cudaMemcpy2D(resultCPU.get(), xlen * sizeof(float), deviceGrayImageOut, outputPitch, xlen * sizeof(float), ylen, cudaMemcpyDeviceToHost)); \
checkCudaErrors(cudaHostUnregister(resultCPU.get())); \
results.push_back(resultCPU); \
break; \
}

namespace processing
{

	/*
	CONVOLUTIONSHAREDSMALL(1,32,16,32,16)
			CONVOLUTIONSHAREDSMALL(3, 32, 8, 31, 7)
			CONVOLUTIONSHAREDSMALL(5, 32, 16, 30, 14)
			CONVOLUTIONSHAREDSMALL(7, 32, 32, 26, 26)
			CONVOLUTIONSHAREDSMALL(9, 32, 32, 24, 24)
			CONVOLUTIONSHAREDSMALL(11, 32, 32, 22, 22)
			CONVOLUTIONSHAREDSMALL(13, 32, 32, 20, 20)
			CONVOLUTIONSHAREDSMALL(15, 32, 16, 27, 11)
	*/

	/*
	// X = 3, Y = 3
	template< typename int FILTER_WIDTH, typename int BLOCK_SIZE_X, typename int BLOCK_SIZE_Y, typename int TILE_SIZE_X, typename int TILE_SIZE_Y>
	__global__ void convolutionGPUSharedSmall(const int numRows, const int numCols, const float * __restrict__ inputImage, float * __restrict__  outputImage, size_t inputPitch, size_t outputPitch)
	{
		const int smallTile = 3;
		float results[smallTile][smallTile] = {0.0};

		__shared__ float shared[BLOCK_SIZE_Y * smallTile][BLOCK_SIZE_X * smallTile];
		const int threadX = threadIdx.x * smallTile;
		const int threadY = threadIdx.y * smallTile;
		int2 absoluteImagePosition;
		absoluteImagePosition.x = IMAD(blockIdx.x, TILE_SIZE_X, threadIdx.x) * smallTile;
		absoluteImagePosition.y = IMAD(blockIdx.y, TILE_SIZE_Y, threadIdx.y) * smallTile;
		float3 firstRow = *(float3 *)(inputImage + IMAD(absoluteImagePosition.y, inputPitch, absoluteImagePosition.x));
		float3 secondRow = *(float3 *)(inputImage + IMAD(absoluteImagePosition.y + 1, inputPitch, absoluteImagePosition.x));
		float3 thirdRow = *(float3 *)(inputImage + IMAD(absoluteImagePosition.y + 2, inputPitch, absoluteImagePosition.x));

		shared[threadY][threadX] = firstRow.x;
		shared[threadY][threadX + 1] = firstRow.y;
		shared[threadY][threadX + 2] = firstRow.z;
		shared[threadY + 1][threadX] = secondRow.x;
		shared[threadY + 1][threadX + 1] = secondRow.y;
		shared[threadY + 1][threadX + 2] = secondRow.z;

		shared[threadY + 2][threadX] = thirdRow.x;
		shared[threadY + 2][threadX + 1] = thirdRow.y;
		shared[threadY + 2][threadX + 2] = thirdRow.z;

		__syncthreads();

		if (threadX < TILE_SIZE_X * smallTile && threadY < TILE_SIZE_Y * smallTile)
		{
			
			float filterValue = 0.0;
#pragma unroll FILTER_WIDTH
			for (int yOffset = 0; yOffset < FILTER_WIDTH; yOffset++)
			{
#pragma unroll FILTER_WIDTH
				for (int xOffset = 0; xOffset < FILTER_WIDTH; xOffset++)
				{
					filterValue = FILTERCUDA[yOffset*FILTER_WIDTH + xOffset];
#pragma unroll smallTile 
					for (int k = 0; k < smallTile; k++)
					{
#pragma unroll smallTile
						for (int l = 0; l < smallTile; l++)
						{
							results[k][l] += filterValue * shared[yOffset + threadY + k][xOffset + threadX + l];
						}
					}
				}
			}
#pragma unroll smallTile 
			for (int k = 0; k < smallTile; k++)
			{
#pragma unroll smallTile
				for (int l = 0; l < smallTile; l++)
				{
					outputImage[IMAD(absoluteImagePosition.y + k, outputPitch, absoluteImagePosition.x) + l] = results[k][l];
				}
			}
		}
	}
	*/

	
	// X = 2 , Y = 2 xx
	//				 xx najlepsie pre 3x3
	template<typename int FILTER_WIDTH, typename int BLOCK_SIZE_X, typename int BLOCK_SIZE_Y, typename int TILE_SIZE_X, typename int TILE_SIZE_Y>
	__global__ void convolutionGPUSharedSmall(const int numRows, const int numCols, const float * __restrict__ inputImage, float * __restrict__  outputImage, size_t inputPitch, size_t outputPitch)
	{
		const int smallTile = 2;
		__shared__ float shared[BLOCK_SIZE_Y * smallTile][BLOCK_SIZE_X * smallTile];
		const int threadX = threadIdx.x * smallTile;
		const int threadY = threadIdx.y * smallTile;
		int2 absoluteImagePosition;
		absoluteImagePosition.x = IMAD(blockIdx.x, TILE_SIZE_X, threadIdx.x) * smallTile;
		absoluteImagePosition.y = IMAD(blockIdx.y, TILE_SIZE_Y, threadIdx.y) * smallTile;
		float2 firstRow =  * (float2 *) (inputImage + IMAD(absoluteImagePosition.y, inputPitch, absoluteImagePosition.x));
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
	

/*
template<typename T, typename int FILTER_WIDTH, typename int BLOCK_SIZE_X, typename int BLOCK_SIZE_Y, typename int TILE_SIZE_X, typename int TILE_SIZE_Y>
__global__ void convolutionGPUSharedSmall(const int numRows, const int numCols, const float * __restrict__ inputImage, T * __restrict__  outputImage, size_t inputPitch, size_t outputPitch)
{
	const int smallTile = 4;
	T results[smallTile] = {0.0};
	__shared__ float shared[BLOCK_SIZE_Y][BLOCK_SIZE_X * smallTile];
	const int threadX = threadIdx.x * smallTile;
	const int threadY = threadIdx.y;
	int2 absoluteImagePosition;
	absoluteImagePosition.x = IMAD(blockIdx.x, TILE_SIZE_X, threadIdx.x) * smallTile;
	absoluteImagePosition.y = IMAD(blockIdx.y, TILE_SIZE_Y, threadIdx.y);
#pragma unroll smallTile
	for (int x = 0; x < smallTile; ++x)
	{
		shared[threadY][threadX + x] = inputImage[IMAD(absoluteImagePosition.y, inputPitch, absoluteImagePosition.x + x)];
	}
	__syncthreads();
	if (threadX < TILE_SIZE_X * smallTile && threadY < TILE_SIZE_Y)
	{
		T filterValue = 0.0;
#pragma unroll FILTER_WIDTH
		for (int yOffset = 0; yOffset < FILTER_WIDTH; yOffset++)
		{
#pragma unroll FILTER_WIDTH
			for (int xOffset = 0; xOffset < FILTER_WIDTH; xOffset++)
			{

				filterValue = FILTERCUDA[yOffset*FILTER_WIDTH + xOffset];
#pragma unroll smallTile
				for (int x = 0; x < smallTile; ++x)
				{
					results[x] += filterValue * shared[yOffset + threadY][xOffset + threadX + x];
				}
			}
		}
#pragma unroll smallTile
		for (int x = 0; x < smallTile; ++x)
		{
			outputImage[IMAD(absoluteImagePosition.y, outputPitch, absoluteImagePosition.x + x)] = results[x];
		}
	}
}
*/

/*
//X = 4, Y = 1
	template<typename T, typename int FILTER_WIDTH, typename int BLOCK_SIZE_X, typename int BLOCK_SIZE_Y, typename int TILE_SIZE_X, typename int TILE_SIZE_Y>
	__global__ void convolutionGPUSharedSmall(const int numRows, const int numCols, const float * __restrict__ inputImage, T * __restrict__  outputImage, size_t inputPitch, size_t outputPitch)
	{
		const int smallTile = 4;
		__shared__ float shared[BLOCK_SIZE_Y][BLOCK_SIZE_X * smallTile];
		const int threadX = threadIdx.x * smallTile;
		const int threadY = threadIdx.y;
		int2 absoluteImagePosition;
		absoluteImagePosition.x = IMAD(blockIdx.x, TILE_SIZE_X, threadIdx.x) * smallTile;
		absoluteImagePosition.y = IMAD(blockIdx.y, TILE_SIZE_Y, threadIdx.y);
		float4 fromGlobal = *((float4 *) (inputImage +IMAD(absoluteImagePosition.y, inputPitch, absoluteImagePosition.x)));
		shared[threadY][threadX] = fromGlobal.x;
		shared[threadY][threadX + 1] = fromGlobal.y;
		shared[threadY][threadX + 2] = fromGlobal.z;
		shared[threadY][threadX + 3] = fromGlobal.w;
		__syncthreads();
		if (threadX < TILE_SIZE_X * smallTile && threadY < TILE_SIZE_Y)
		{
			T result1 = 0.0; 
			T result2 = 0.0; 
			T result3 = 0.0; 
			T result4 = 0.0;
			T filterValue = 0.0;
#pragma unroll FILTER_WIDTH
			for (int yOffset = 0; yOffset < FILTER_WIDTH; yOffset++)
			{
#pragma unroll FILTER_WIDTH
				for (int xOffset = 0; xOffset < FILTER_WIDTH; xOffset++)
				{
					filterValue = FILTERCUDA[yOffset*FILTER_WIDTH + xOffset];
					result1 += filterValue * shared[yOffset + threadY][xOffset + threadX];
					result2 += filterValue * shared[yOffset + threadY][xOffset + threadX + 1];
					result3 += filterValue * shared[yOffset + threadY][xOffset + threadX + 2];
					result4 += filterValue * shared[yOffset + threadY][xOffset + threadX + 3];
				}
			}
			outputImage[IMAD(absoluteImagePosition.y, outputPitch, absoluteImagePosition.x)] = result1;
			outputImage[IMAD(absoluteImagePosition.y, outputPitch, absoluteImagePosition.x + 1)] = result2;
			outputImage[IMAD(absoluteImagePosition.y, outputPitch, absoluteImagePosition.x + 2)] = result3;
			outputImage[IMAD(absoluteImagePosition.y, outputPitch, absoluteImagePosition.x + 3)] = result4;
		}
	}
	*/

	/*
	template<typename T, typename int FILTER_WIDTH, typename int BLOCK_SIZE_X, typename int BLOCK_SIZE_Y, typename int TILE_SIZE_X, typename int TILE_SIZE_Y>
	__global__ void convolutionGPUSharedSmall(const int numRows, const int numCols, const uchar * __restrict__ inputImage, T * __restrict__  outputImage, size_t inputPitch, size_t outputPitch)
	{
		const int smallTile = 2;
		__shared__ float shared[BLOCK_SIZE_Y][BLOCK_SIZE_X * smallTile];
		const int threadX = threadIdx.x * smallTile;
		const int threadY = threadIdx.y;
		int2 absoluteImagePosition;
		absoluteImagePosition.x = IMAD(blockIdx.x, TILE_SIZE_X, threadIdx.x) * smallTile;
		absoluteImagePosition.y = IMAD(blockIdx.y, TILE_SIZE_Y, threadIdx.y);
#pragma unroll smallTile
		for (int x = 0; x < smallTile; ++x)
		{
			shared[threadY][threadX + x] = inputImage[IMAD(absoluteImagePosition.y, inputPitch, absoluteImagePosition.x + x)];
		}
		__syncthreads();
		if (threadX < TILE_SIZE_X * smallTile && threadY < TILE_SIZE_Y)
		{
			T result1 = 0.0; //xx
			T result2 = 0.0; //x0
			T filterValue = 0.0;
#pragma unroll FILTER_WIDTH
				for (int yOffset = 0; yOffset < FILTER_WIDTH; yOffset++)
				{
#pragma unroll FILTER_WIDTH
					for (int xOffset = 0; xOffset < FILTER_WIDTH; xOffset++)
					{
						filterValue = FILTERCUDA[yOffset*FILTER_WIDTH + xOffset];
						result1 += filterValue * shared[yOffset + threadY][xOffset + threadX];
						result2 += filterValue * shared[yOffset + threadY][xOffset + threadX + 1];
					}
				}
				outputImage[IMAD(absoluteImagePosition.y, outputPitch, absoluteImagePosition.x)] = result1;
				outputImage[IMAD(absoluteImagePosition.y, outputPitch, absoluteImagePosition.x + 1)] = result2;
		}
	}
	*/

	/*X = 1, Y = 1

		template<typename T, typename int FILTER_WIDTH, typename int BLOCK_SIZE_X, typename int BLOCK_SIZE_Y, typename int TILE_SIZE_X, typename int TILE_SIZE_Y>
	__global__ void convolutionGPUSharedSmall(const int numRows, const int numCols, const uchar * __restrict__ inputImage, T * __restrict__  outputImage, int inputPitch, int outputPitch)
	{
		__shared__ float shared[BLOCK_SIZE_Y][BLOCK_SIZE_X];
		int threadX = threadIdx.x;
		int threadY = threadIdx.y;
		int2 absoluteImagePosition;
		absoluteImagePosition.x = IMAD(blockIdx.x, TILE_SIZE_X, threadX);
		absoluteImagePosition.y = IMAD(blockIdx.y, TILE_SIZE_Y, threadY);
		shared[threadY][threadX] = inputImage[IMAD(absoluteImagePosition.y, inputPitch, absoluteImagePosition.x)];
		__syncthreads();

		if (threadX < TILE_SIZE_X && threadY < TILE_SIZE_Y)
		{
			T result = 0.0;
			//threadX += (FILTER_WIDTH / 2);
			//threadY += (FILTER_WIDTH / 2);
#pragma unroll FILTER_WIDTH
			for (int yOffset = 0; yOffset < FILTER_WIDTH; yOffset++)
			{
#pragma unroll FILTER_WIDTH
				for (int xOffset = 0; xOffset < FILTER_WIDTH; xOffset++)
				{
					result += FILTERCUDA[yOffset*FILTER_WIDTH + xOffset] * shared[yOffset + threadY][xOffset + threadX];
				}
			}
			outputImage[IMAD(absoluteImagePosition.y, outputPitch, absoluteImagePosition.x)] = result;
		}
	}
	*/

	KernelSharedForSmall::KernelSharedForSmall()
	{}


	void KernelSharedForSmall::run(ImageFactory& image, vector<shared_ptr<AbstractFilter>>& filters, vector<shared_ptr<float>>& results)
	{
		const short MAX_BLOCK_SIZE_X = 64;
		const short MAX_BLOCK_SIZE_Y = 64;
		const short MAX_TILE_SIZE_X = 62;
		const short MAX_TILE_SIZE_Y = 62;
		const short MAX_SMALL_TILE_DIMENION_X = 2;
		const short MAX_SMALL_TILE_DIMENION_Y = 2;
		float* deviceGrayImageOut = 0;
		size_t outputPitch = 0;
		checkCudaErrors(cudaMallocPitch<float>(&deviceGrayImageOut, &outputPitch, (image.getNumCols() + MAX_BLOCK_SIZE_X * 2)* sizeof(float) , (image.getNumRows() + MAX_BLOCK_SIZE_Y* 2)));
		float* hostGrayImage = image.getInputGrayPointerFloat();

		// memory allocation
		int colsForGridX = CEIL(image.getNumCols(), MAX_SMALL_TILE_DIMENION_X);
		int rowsForGridY = CEIL(image.getNumRows(), MAX_SMALL_TILE_DIMENION_Y);

		float* deviceGrayImageIn;
		size_t inputPitch = 0;

		checkCudaErrors(cudaMallocPitch<float>(&deviceGrayImageIn, &inputPitch, (image.getNumCols() + MAX_BLOCK_SIZE_X * MAX_SMALL_TILE_DIMENION_X) * sizeof(float), image.getNumRows() + MAX_BLOCK_SIZE_Y* MAX_SMALL_TILE_DIMENION_Y));
		checkCudaErrors( cudaDeviceSynchronize());
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
		cudaFree(deviceGrayImageIn);
		cudaFree(deviceGrayImageOut);
	}

}

