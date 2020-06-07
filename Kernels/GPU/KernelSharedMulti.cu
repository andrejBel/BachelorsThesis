#include "KernelSharedMulti.h"


#include <cuda.h>
#include <helper_cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include "opencv2/core/utility.hpp"

#include "MemoryPoolPinned.h"
#include <vector>
#include <cstdio>
#include <cmath>
#include <iostream>

#include <algorithm>
#include <type_traits>
#include <utility>

#define MUL(a, b) __mul24(a, b)
#define IMAD(a, b, c) ( __mul24((a), (b)) + (c) )
#define CEIL(a, b) ((a + b - 1) / b)
#define ROUNDUP(a, b) (((a + b - 1) / b) * b)

using namespace std;
namespace processing
{
#define  CONVOLUTIONSHAREDMULTI(FILTERWIDTH, BLOCKSIZEX, BLOCKSIZEY, TILESIZEX, TILESIZEY) \
					case FILTERWIDTH: \
					{ \
						const short MAX_SMALL_TILE_DIMENION_X = 2;\
						const short MAX_SMALL_TILE_DIMENION_Y = 2;\
						int colsForGridX = CEIL(numCols, MAX_SMALL_TILE_DIMENION_X); \
						int rowsForGridY = CEIL(numRows, MAX_SMALL_TILE_DIMENION_Y); \
						const int FILTER_WIDTH = FILTERWIDTH; \
						const int BLOCK_SIZE_X = BLOCKSIZEX; \
						const int BLOCK_SIZE_Y = BLOCKSIZEY; \
						const int TILE_SIZE_X = TILESIZEX; \
						const int TILE_SIZE_Y = TILESIZEY; \
						const dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y); \
						const dim3 gridSize((colsForGridX + TILE_SIZE_X - 1) / TILE_SIZE_X, (rowsForGridY + TILE_SIZE_Y - 1) / TILE_SIZE_Y, 1); \
						if (makezeros)\
						{\
							convolutionGPUSharedIncompleteBlock< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, true> << <gridSize, blockSize >> > (inputImagesDevice[k], deviceGrayImageOut, pitchInput / sizeof(float), pitchOutput / sizeof(float), k);\
						}\
						else\
						{\
							convolutionGPUSharedIncompleteBlock< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, false> << <gridSize, blockSize >> > (inputImagesDevice[k], deviceGrayImageOut, pitchInput / sizeof(float), pitchOutput / sizeof(float), k);\
						} \
						break; \
					}

#define CONVOLUTIONSHAREDMULTIFULL(FILTERWIDTH, BLOCKSIZEX, BLOCKSIZEY) \
					case FILTERWIDTH: \
					{ \
						const short MAX_SMALL_TILE_DIMENION_X = 3;\
						const short MAX_SMALL_TILE_DIMENION_Y = 3;\
						int colsForGridX = CEIL(numCols, MAX_SMALL_TILE_DIMENION_X); \
						int rowsForGridY = CEIL(numRows, MAX_SMALL_TILE_DIMENION_Y); \
						const int FILTER_WIDTH = FILTERWIDTH; \
						const int BLOCK_SIZE_X = BLOCKSIZEX; \
						const int BLOCK_SIZE_Y = BLOCKSIZEY; \
						const int TILE_SIZE_X = 3; \
						const int TILE_SIZE_Y = 3; \
						const dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y); \
						const dim3 gridSize((colsForGridX + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (rowsForGridY + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y, 1); \
						if (makezeros) \
						{\
							convolutionGPUSharedFullBlock< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, true> << <gridSize, blockSize >> > (inputImagesDevice[k], deviceGrayImageOut, pitchInput / sizeof(float), pitchOutput / sizeof(float), k); \
						}\
						else \
						{\
							convolutionGPUSharedFullBlock< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, false> << <gridSize, blockSize >> > (inputImagesDevice[k], deviceGrayImageOut, pitchInput / sizeof(float), pitchOutput / sizeof(float), k); \
						}\
						break; \
					}


	template<typename int FILTER_WIDTH, typename int BLOCK_SIZE_X, typename int BLOCK_SIZE_Y, typename int TILE_SIZE_X, typename int TILE_SIZE_Y, typename bool MAKEZEROS, typename int MAX_SMALL_TILE_DIMENION_X = 2, typename int MAX_SMALL_TILE_DIMENION_Y = 2>
	__global__ void convolutionGPUSharedIncompleteBlock(float * __restrict__ inputImage, float * __restrict__  outputImage, int inputPitch, int outputPitch, const short filterIndex)
	{
		__shared__ float shared[BLOCK_SIZE_Y * MAX_SMALL_TILE_DIMENION_Y][BLOCK_SIZE_X * MAX_SMALL_TILE_DIMENION_X];
		int threadX = MUL(threadIdx.x, MAX_SMALL_TILE_DIMENION_X);
		int threadY = MUL(threadIdx.y, MAX_SMALL_TILE_DIMENION_Y);
		int2 absoluteImagePosition;
		absoluteImagePosition.x = IMAD(blockIdx.x, TILE_SIZE_X, threadIdx.x) * MAX_SMALL_TILE_DIMENION_X;
		absoluteImagePosition.y = IMAD(blockIdx.y, TILE_SIZE_Y, threadIdx.y) * MAX_SMALL_TILE_DIMENION_Y;

#pragma unroll MAX_SMALL_TILE_DIMENION_Y
		for (int i = 0; i < MAX_SMALL_TILE_DIMENION_Y; i++)
		{
			* ((float2 *)&shared[threadY + i][threadX]) = *(float2 *)(inputImage + IMAD(absoluteImagePosition.y + i, inputPitch, absoluteImagePosition.x));
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
					filterValue = FILTERCUDA[yOffset*FILTER_WIDTH + xOffset + FILTER_WIDTH * FILTER_WIDTH * filterIndex];
#pragma unroll MAX_SMALL_TILE_DIMENION_Y
					for (int i = 0; i < MAX_SMALL_TILE_DIMENION_Y; i++)
					{
#pragma unroll MAX_SMALL_TILE_DIMENION_X
						for (int j = 0; j < MAX_SMALL_TILE_DIMENION_X; j++)
						{
							results[IMAD(i, MAX_SMALL_TILE_DIMENION_Y, j)] += filterValue * shared[yOffset + threadY + i][xOffset + threadX + j];
						}

					}
				}
			}
			/*
#pragma unroll MAX_SMALL_TILE_DIMENION_Y
			for (int i = 0; i < MAX_SMALL_TILE_DIMENION_Y; i++)
			{
				//* ((float2 *)(outputImage + IMAD(absoluteImagePosition.y + i, outputPitch, absoluteImagePosition.x))) = *((float2 *)(&results[MUL(i, MAX_SMALL_TILE_DIMENION_Y)]));

#pragma unroll MAX_SMALL_TILE_DIMENION_X
				for (int j = 0; j < MAX_SMALL_TILE_DIMENION_X; j++)
				{
					outputImage[IMAD(absoluteImagePosition.y + i, outputPitch, absoluteImagePosition.x + j)] += results[i *MAX_SMALL_TILE_DIMENION_Y + j];
				}
			}
			*/
			if (MAKEZEROS)
			{
				*((float2 *)(outputImage + IMAD(absoluteImagePosition.y, outputPitch, absoluteImagePosition.x))) = *((float2*)&results[0]);
				*((float2 *)(outputImage + IMAD(absoluteImagePosition.y + 1, outputPitch, absoluteImagePosition.x))) = *((float2*)&results[2]);
			}
			else
			{
				float2 fromGlobal = *((float2 *)(outputImage + IMAD(absoluteImagePosition.y, outputPitch, absoluteImagePosition.x)));
				fromGlobal.x += results[0];
				fromGlobal.y += results[1];
				*((float2 *)(outputImage + IMAD(absoluteImagePosition.y, outputPitch, absoluteImagePosition.x))) = fromGlobal;
				fromGlobal = *((float2 *)(outputImage + IMAD(absoluteImagePosition.y + 1, outputPitch, absoluteImagePosition.x)));
				fromGlobal.x += results[2];
				fromGlobal.y += results[3];
				*((float2 *)(outputImage + IMAD(absoluteImagePosition.y + 1, outputPitch, absoluteImagePosition.x))) = fromGlobal;
			}
		}
	}

	template<typename int FILTER_WIDTH, typename int BLOCK_SIZE_X, typename int BLOCK_SIZE_Y, typename int TILE_SIZE_X, typename int TILE_SIZE_Y, typename bool MAKEZEROS>
	__global__ void convolutionGPUSharedFullBlock(float * __restrict__ inputImage, float *__restrict__ outputImage, const int inputPitch, const int outputPitch, const short filterIndex)
	{
		float results[TILE_SIZE_X * TILE_SIZE_Y];
		int2 absoluteImagePosition;
		absoluteImagePosition.x = (blockIdx.x *  blockDim.x + threadIdx.x) * TILE_SIZE_X;
		absoluteImagePosition.y = (blockIdx.y * blockDim.y + threadIdx.y) * TILE_SIZE_Y;
		int2 positionShared;
		positionShared.x = blockIdx.x * BLOCK_SIZE_X * TILE_SIZE_X;
		positionShared.y = blockIdx.y * BLOCK_SIZE_Y * TILE_SIZE_Y;
		__shared__ float shared[BLOCK_SIZE_Y*TILE_SIZE_Y + (FILTER_WIDTH - 1) + TILE_SIZE_Y][BLOCK_SIZE_X *TILE_SIZE_X + (FILTER_WIDTH - 1) + TILE_SIZE_X];
		int threadX = threadIdx.x * TILE_SIZE_X;
		int threadY = threadIdx.y * TILE_SIZE_Y;
		for (int j = threadY; j < FILTER_WIDTH - 1 + BLOCK_SIZE_Y * TILE_SIZE_Y; j += BLOCK_SIZE_Y * TILE_SIZE_Y)
		{
			for (int i = threadX; i < FILTER_WIDTH - 1 + BLOCK_SIZE_X * TILE_SIZE_X; i += BLOCK_SIZE_X * TILE_SIZE_X)
			{
#pragma unroll TILE_SIZE_Y
				for (int k = 0; k < TILE_SIZE_Y; k++)
				{

					*((float3 *)&shared[j + k][i]) = *(float3 *)(inputImage + IMAD(positionShared.y + j + k, inputPitch, positionShared.x + i));
				}
			}
		}
		__syncthreads();
#pragma unroll TILE_SIZE_Y
		for (int k = 0; k < TILE_SIZE_Y; k++)
		{
#pragma unroll TILE_SIZE_X
			for (int l = 0; l < TILE_SIZE_X; l++)
			{
				results[k * TILE_SIZE_Y + l] = 0.0;
			}
		}
		
		float filterValue;
#pragma unroll FILTER_WIDTH
		for (int yOffset = 0; yOffset < FILTER_WIDTH; yOffset++)
		{
#pragma unroll FILTER_WIDTH
			for (int xOffset = 0; xOffset < FILTER_WIDTH; xOffset++)
			{
				filterValue = FILTERCUDA[yOffset*FILTER_WIDTH + xOffset + FILTER_WIDTH * FILTER_WIDTH * filterIndex];
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
		if (MAKEZEROS)
		{
			*((float3 *)(outputImage + IMAD(absoluteImagePosition.y, outputPitch, absoluteImagePosition.x))) = *((float3*)&results[0]);
			*((float3 *)(outputImage + IMAD(absoluteImagePosition.y + 1, outputPitch, absoluteImagePosition.x))) = *((float3*)&results[3]);
			*((float3 *)(outputImage + IMAD(absoluteImagePosition.y + 2, outputPitch, absoluteImagePosition.x))) = *((float3*)&results[6]);
		}
		else
		{
			float3 fromGlobal = *((float3 *)(outputImage + IMAD(absoluteImagePosition.y, outputPitch, absoluteImagePosition.x)));
			fromGlobal.x += results[0];
			fromGlobal.y += results[1];
			fromGlobal.z += results[2];
			*((float3 *)(outputImage + IMAD(absoluteImagePosition.y, outputPitch, absoluteImagePosition.x))) = fromGlobal;
			fromGlobal = *((float3 *)(outputImage + IMAD(absoluteImagePosition.y + 1, outputPitch, absoluteImagePosition.x)));
			fromGlobal.x += results[3];
			fromGlobal.y += results[4];
			fromGlobal.z += results[5];
			*((float3 *)(outputImage + IMAD(absoluteImagePosition.y + 1, outputPitch, absoluteImagePosition.x))) = fromGlobal;
			fromGlobal = *((float3 *)(outputImage + IMAD(absoluteImagePosition.y + 2, outputPitch, absoluteImagePosition.x)));
			fromGlobal.x += results[6];
			fromGlobal.y += results[7];
			fromGlobal.z += results[8];
			*((float3 *)(outputImage + IMAD(absoluteImagePosition.y + 2, outputPitch, absoluteImagePosition.x))) = fromGlobal;
		}

	}

	KernelSharedMulti::KernelSharedMulti() : MultiRunnable()
	{
	}

	void KernelSharedMulti::run(vector<shared_ptr<ImageFactory>>& images, vector<vector<shared_ptr<Filter>>>& filters, vector<shared_ptr<float>>& results)
	{
		
		int maxImageWidth = 0;
		int maxImageHeight = 0;
		for_each(images.begin(), images.end(), [&maxImageWidth, &maxImageHeight](shared_ptr<ImageFactory> image)
		{
			if (image->getNumCols() > maxImageWidth)
			{
				maxImageWidth = image->getNumCols();
			}
			if (image->getNumRows() > maxImageHeight)
			{
				maxImageHeight = image->getNumRows();
			}
		});
		MemoryPoolPitched::getMemoryPoolPitchedForInput().realoc(maxImageWidth, maxImageHeight);
		MemoryPoolPitched::getMemoryPoolPitchedForOutput().realoc(maxImageWidth, maxImageHeight);

		size_t pitchInput = MemoryPoolPitched::getMemoryPoolPitchedForInput().getPitch();
		size_t pitchOutput = MemoryPoolPitched::getMemoryPoolPitchedForOutput().getPitch();
		int numCols = images[0]->getNumCols(); //x
		int numRows = images[0]->getNumRows(); //y

		vector<float *> inputImagesDevice;
		for (int i = 0; i < PITCHED_MEMORY_BUFFER_SIZE_INPUT; i++)
		{
			inputImagesDevice.push_back(MemoryPoolPitched::getMemoryPoolPitchedForInput().getMemory()[i]);
		}
		for (int i = 0; i < PITCHED_MEMORY_BUFFER_SIZE_OUTPUT - 1; i++)
		{
			inputImagesDevice.push_back(MemoryPoolPitched::getMemoryPoolPitchedForOutput().getMemory()[i]);
		}
		float * deviceGrayImageOut = MemoryPoolPitched::getMemoryPoolPitchedForOutput().getMemory()[PITCHED_MEMORY_BUFFER_SIZE_OUTPUT - 1];



		int imageSize = static_cast<int>(images.size());
		size_t filterGroupCount = filters.size();
		vector<vector<shared_ptr<float>>> partialResults(filterGroupCount);



		for (int i = 0; i < imageSize; i += inputImagesDevice.size())
		{
			int startOfImages = i;
			int endOfImages = std::min(i + static_cast<int>(inputImagesDevice.size()) - 1, imageSize - 1);
			int usedImages = endOfImages - startOfImages + 1;

			for (int indexImages = startOfImages, indexDeviceInput = 0; indexImages <= endOfImages; ++indexImages, ++indexDeviceInput)
			{
				checkCudaErrors(cudaMemcpy2D(inputImagesDevice[indexDeviceInput], pitchInput, images[indexImages]->getInputGrayPointerFloat(), numCols * sizeof(float), numCols * sizeof(float), numRows, cudaMemcpyHostToDevice));
			}

			for (size_t j = 0; j < filterGroupCount; ++j)
			{
				//checkCudaErrors(cudaMemset2D(deviceGrayImageOut, pitchOutput, 0, numCols * sizeof(float), numRows));
				vector<shared_ptr<Filter>>& groupFilters = filters[j];
				int filterWidth = groupFilters[0]->getWidth();
				int sizeOfFilter = filterWidth* filterWidth;
				shared_ptr<float> memoryForFilters = shared_ptr<float>(new float[sizeOfFilter * usedImages], [](float * ptr) { delete[] ptr; });
				for (int offset = 0, index = startOfImages; index <= endOfImages; ++index, offset += (filterWidth * filterWidth))
				{
					std::copy(groupFilters[index]->getFilter(), groupFilters[index]->getFilter() + sizeOfFilter, memoryForFilters.get() + offset);
				}
				checkCudaErrors(cudaMemcpyToSymbol(FILTERCUDA, memoryForFilters.get(), sizeof(float) * sizeOfFilter * usedImages));
				int xlen = numCols - (filterWidth - 1);
				int ylen = numRows - (filterWidth - 1);
				
				for (int k = 0; k < usedImages; k++)
				{
					bool makezeros = false;
					if (k == 0) 
					{
						makezeros = true;
					}
					switch (filterWidth)
					{
						/*
						CONVOLUTIONSHAREDMULTI(1, 32, 16, 32, 16);
						CONVOLUTIONSHAREDMULTI(3, 32, 16, 31, 15);
						CONVOLUTIONSHAREDMULTI(5, 32, 16, 30, 14);
						CONVOLUTIONSHAREDMULTI(7, 32, 32, 29, 29)
						CONVOLUTIONSHAREDMULTI(9, 32, 32, 28, 28)
						CONVOLUTIONSHAREDMULTI(11, 32, 32, 27, 27)
						CONVOLUTIONSHAREDMULTI(13, 32, 32, 26, 26)
						CONVOLUTIONSHAREDMULTI(15, 32, 32, 25, 25)
							*/
							
						CONVOLUTIONSHAREDMULTI(1, 32, 16, 32, 16)
						CONVOLUTIONSHAREDMULTI(3, 32, 16, 31, 15)
						CONVOLUTIONSHAREDMULTI(5, 32, 16, 30, 14)
						CONVOLUTIONSHAREDMULTI(7, 32, 32, 29, 29)
						CONVOLUTIONSHAREDMULTIFULL(9, 32, 8)
						CONVOLUTIONSHAREDMULTIFULL(11, 32, 8)
						CONVOLUTIONSHAREDMULTIFULL(13, 32, 6);
						CONVOLUTIONSHAREDMULTIFULL(15, 32, 8);
						
					default:
						std::cerr << "Filter with width: " << filterWidth << " not supported!" << endl;
						break;
					}
				}
				shared_ptr<float> resultCPU = MemoryPoolPinned::getMemoryPoolPinnedForOutput().acquireMemory(xlen*ylen, true);
				checkCudaErrors(cudaMemcpy2D(resultCPU.get(), xlen * sizeof(float), deviceGrayImageOut, pitchOutput, xlen * sizeof(float), ylen, cudaMemcpyDeviceToHost));
				checkCudaErrors(cudaDeviceSynchronize());
				partialResults[j].push_back(resultCPU);
			}
		}

		for (size_t j = 0; j < filterGroupCount; ++j)
		{
			int filterWidth = filters[j][0]->getWidth();
			int xlen = numCols - (filterWidth - 1);
			int ylen = numRows - (filterWidth - 1);
			size_t range = xlen*ylen;
			shared_ptr<float> result = partialResults[j][0];
			float* resultPointer = result.get();
			for (int i = 1; i < partialResults[j].size(); ++i)
			{
				float * nextResult = partialResults[j][i].get();
				for (size_t k = 0; k < range; ++k)
				{
					resultPointer[k] += nextResult[k];
				}
			}
			results.push_back(result);
		}

	}

}