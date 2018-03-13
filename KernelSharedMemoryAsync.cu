#include "KernelSharedMemoryAsync.h"

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
#include <type_traits>

using namespace std;


#define CONVOLUTIONSHAREDASYNC(FILTER_W, BLOCK_S, TILE_S)\
case FILTER_W:\
{\
	Filter<T, FILTER_W> * ptr = (Filter<T, FILTER_W> *) (deviceFilters.get() + offset);\
	cudaMemcpyToSymbol(FILTERCUDA, ptr->getFilter(), sizeof(float)*FILTER_W*FILTER_W);\
	const int BLOCK_SIZE = BLOCK_S;\
	const int FILTER_WIDTH = FILTER_W;\
	const int TILE_SIZE = TILE_S;\
	static_assert(BLOCK_SIZE - TILE_SIZE >= (FILTER_WIDTH / 2), "Wrong block and tile size, BLOCKSIZE - TILESIZE >= (FILTERWIDTH - 1)");\
	const dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);\
	const dim3 gridSize((colsForGridX + TILE_SIZE - 1) / TILE_SIZE, (rowsForGridY + TILE_SIZE - 1) / TILE_SIZE, 1);\
	convolutionGPUSharedAsync<T, FILTER_WIDTH, BLOCK_SIZE, TILE_SIZE> << <gridSize, blockSize >> >(image.getNumRows(), image.getNumCols(), deviceGrayImageIn.get(), deviceGrayImageOut.get());\
	break;\
}

#define IMAD(a, b, c) ( __mul24((a), (b)) + (c) )



namespace processing
{
	
	

	template<typename T, typename int FILTER_WIDTH, typename int BLOCK_SIZE, typename int TILE_SIZE>
	__global__ void convolutionGPUSharedAsync(const int numRows, const int numCols,const uchar * __restrict__ inputImage, T * __restrict__  outputImage, size_t inputPitch, size_t outputPitch)
	{
		//if ((IMAD(blockIdx.y, TILE_SIZE, threadIdx.y)*numCols  + IMAD(blockIdx.x, TILE_SIZE, threadIdx.x)) == 1)
		//{
			
			//printFromKernel("Filter width", FILTER_WIDTH);
			//printFromKernel("TILE size", TILE_SIZE);
			//printFromKernel("BLOCK size", BLOCK_SIZE);
			__shared__ float shared[BLOCK_SIZE * 2][BLOCK_SIZE * 2];
			const int smallTile = 2;
			const int threadX = threadIdx.x * smallTile;
			const int threadY = threadIdx.y * smallTile;
			int2 absoluteImagePosition;
			absoluteImagePosition.x = IMAD(blockIdx.x, TILE_SIZE , threadIdx.x) * smallTile;
			absoluteImagePosition.y = IMAD(blockIdx.y, TILE_SIZE , threadIdx.y) * smallTile;
			//printFromKernel("Absolute image position x", absoluteImagePosition.x);
			//printFromKernel("Absolute image position y", absoluteImagePosition.y);
			
			int2 sharedPosition[smallTile][smallTile];
			#pragma unroll smallTile
			for (int y = 0; y < smallTile; ++y)
			{
				#pragma unroll smallTile
				for (int x = 0; x < smallTile; ++x)
				{
					sharedPosition[y][x].x = absoluteImagePosition.x - (FILTER_WIDTH / 2) + x;
					sharedPosition[y][x].y = absoluteImagePosition.y - (FILTER_WIDTH / 2) + y;
					sharedPosition[y][x].x = min(max(sharedPosition[y][x].x, 0), numCols - 1);
					sharedPosition[y][x].y = min(max(sharedPosition[y][x].y, 0), numRows - 1);
					shared[threadY + y][threadX + x] = inputImage[IMAD(sharedPosition[y][x].y, inputPitch, sharedPosition[y][x].x)];
				}
			}
			//printFromKernel("shared positon x 1", sharedPosition1.x);
			//printFromKernel("shared positon y 1", sharedPosition1.y);

			//printFromKernel("shared positon x 2", sharedPosition2.x);
			//printFromKernel("shared positon y 2", sharedPosition2.y);
			__syncthreads();

			if (threadX < TILE_SIZE * 2 && threadY < TILE_SIZE * 2)
			{
				T result1 = 0.0; //00
				T result2 = 0.0; //01
				T result3 = 0.0; //10
				T result4 = 0.0; //11
				T filterValue = 0.0;
				if ((absoluteImagePosition.x + 1) < numCols && (absoluteImagePosition.y + 1) <  numRows) // all from small tile xx
				{																						//                      xx
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
				else if ((absoluteImagePosition.x + 1) < numCols && (absoluteImagePosition.y) <  numRows) // xx
				{																						  //  00				
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
				else if ((absoluteImagePosition.x) < numCols && (absoluteImagePosition.y + 1) <  numRows)// x0
				{																						 //  x0	
				#pragma unroll FILTER_WIDTH
				for (int yOffset = 0; yOffset < FILTER_WIDTH; yOffset++)
				{
				#pragma unroll FILTER_WIDTH
					for (int xOffset = 0; xOffset < FILTER_WIDTH; xOffset++)
					{
						filterValue = FILTERCUDA[yOffset*FILTER_WIDTH + xOffset];
						result1 += filterValue * shared[yOffset + threadY][xOffset + threadX];
						result3 += filterValue * shared[yOffset + threadY + 1][xOffset + threadX];
					}
				}
				outputImage[IMAD(absoluteImagePosition.y, outputPitch, absoluteImagePosition.x)] = result1;
				outputImage[IMAD(absoluteImagePosition.y + 1, outputPitch, absoluteImagePosition.x)] = result3;
				}
				else if (absoluteImagePosition.x < numCols && absoluteImagePosition.y <  numRows)  // x0
				{																					// 00
				#pragma unroll FILTER_WIDTH
				for (int yOffset = 0; yOffset < FILTER_WIDTH; yOffset++)
				{
				#pragma unroll FILTER_WIDTH
					for (int xOffset = 0; xOffset < FILTER_WIDTH; xOffset++)
					{
						result1 += FILTERCUDA[yOffset*FILTER_WIDTH + xOffset] * shared[yOffset + threadY][xOffset + threadX];
					}
				}
					outputImage[IMAD(absoluteImagePosition.y, outputPitch, absoluteImagePosition.x)] = result1;
				}
			}
			//if ((IMAD(blockIdx.y, TILE_SIZE, threadIdx.y)*numCols + IMAD(blockIdx.x, TILE_SIZE, threadIdx.x)) == 0)	{}
	}

	template<typename T>
	KernelSharedMemoryAsync<T>::KernelSharedMemoryAsync()
	{}

	template<typename T>
	void KernelSharedMemoryAsync<T>::run(ImageFactory& image, vector<shared_ptr<AbstractFilter<T>>>& filters, vector<shared_ptr<T>>& results)
	{

		T* deviceGrayImageOut = 0;
		size_t outputPitch = 0;
		checkCudaErrors(cudaMallocPitch<T>(&deviceGrayImageOut, &outputPitch, image.getNumCols() * sizeof(T), image.getNumRows()));
		uchar* hostGrayImage = image.getInputGrayPointer();
		
		// memory allocation
		int colsForGridX = (image.getNumCols() + 1) / 2;
		int rowsForGridY = (image.getNumRows() + 1) / 2;



		uchar* deviceGrayImageIn;
		size_t inputPitch = 0;

		checkCudaErrors(cudaMallocPitch<uchar>(&deviceGrayImageIn, &inputPitch, image.getNumCols(), image.getNumRows()));
		cudaDeviceSynchronize();
		checkCudaErrors(cudaMemcpy2D(deviceGrayImageIn, inputPitch, hostGrayImage, image.getNumCols(), image.getNumCols()* sizeof(uchar), image.getNumRows(), cudaMemcpyHostToDevice));


		for (auto& filter : filters)
		{
			switch (filter->getWidth())
			{
			case 3:
			{
				checkCudaErrors(cudaMemcpyToSymbol(FILTERCUDA, ((Filter<T,3> *) filter.get())->getFilter(), sizeof(T) * 3 * 3));
				const int BLOCK_SIZE = 16;
				const int FILTER_WIDTH = 3;
				const int TILE_SIZE = 15;
				static_assert(BLOCK_SIZE - TILE_SIZE >= (FILTER_WIDTH / 2), "Wrong block and tile size, BLOCKSIZE - TILESIZE >= (FILTERWIDTH - 1)");
				const dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
				const dim3 gridSize((colsForGridX + TILE_SIZE - 1) / TILE_SIZE, (rowsForGridY + TILE_SIZE - 1) / TILE_SIZE, 1);
				convolutionGPUSharedAsync<T, FILTER_WIDTH, BLOCK_SIZE, TILE_SIZE> << <gridSize, blockSize >> >(image.getNumRows(), image.getNumCols(), deviceGrayImageIn, deviceGrayImageOut, inputPitch, outputPitch / sizeof(T));
				break;
			}
			case 5:
			{
				checkCudaErrors(cudaMemcpyToSymbol(FILTERCUDA, ((Filter<T, 5> *) filter.get())->getFilter(), sizeof(T) * 5 * 5));
				const int BLOCK_SIZE = 16;
				const int FILTER_WIDTH = 5;
				const int TILE_SIZE = 14;
				static_assert(BLOCK_SIZE - TILE_SIZE >= (FILTER_WIDTH / 2), "Wrong block and tile size, BLOCKSIZE - TILESIZE >= (FILTERWIDTH - 1)");
				const dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
				const dim3 gridSize((colsForGridX + TILE_SIZE - 1) / TILE_SIZE, (rowsForGridY + TILE_SIZE - 1) / TILE_SIZE, 1);
				convolutionGPUSharedAsync<T, FILTER_WIDTH, BLOCK_SIZE, TILE_SIZE> << <gridSize, blockSize >> >(image.getNumRows(), image.getNumCols(), deviceGrayImageIn, deviceGrayImageOut, inputPitch, outputPitch / sizeof(T));
				break;
			}
			case 15:
			{
				checkCudaErrors(cudaMemcpyToSymbol(FILTERCUDA, ((Filter<T, 15> *) filter.get())->getFilter(), sizeof(float)* 15 * 15));
				const int BLOCK_SIZE = 32; 
				const int FILTER_WIDTH = 15;
				const int TILE_SIZE = 25; 
				static_assert(BLOCK_SIZE - TILE_SIZE >= (FILTER_WIDTH / 2), "Wrong block and tile size, BLOCKSIZE - TILESIZE >= (FILTERWIDTH - 1)"); 
				const dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE); 
				const dim3 gridSize((colsForGridX + TILE_SIZE - 1) / TILE_SIZE, (rowsForGridY + TILE_SIZE - 1) / TILE_SIZE, 1); 
				convolutionGPUSharedAsync<T, FILTER_WIDTH, BLOCK_SIZE, TILE_SIZE> << <gridSize, blockSize >> >(image.getNumRows(), image.getNumCols(), deviceGrayImageIn, deviceGrayImageOut, inputPitch, outputPitch / sizeof(T));
				break; 
			}
					
				//CONVOLUTIONSHAREDASYNC(1, 32, 32)
				//CONVOLUTIONSHAREDASYNC(3, 32, 31)
				//CONVOLUTIONSHAREDASYNC(5, 32, 30)
				//CONVOLUTIONSHAREDASYNC(7, 32, 29)
				//CONVOLUTIONSHAREDASYNC(9, 32, 28)
				//CONVOLUTIONSHAREDASYNC(11, 32, 27)
				//CONVOLUTIONSHAREDASYNC(13, 32, 26)
				//CONVOLUTIONSHAREDASYNC(15, 32, 25)
				
			
			default:
				std::cerr << "Filter with width: " << filter->getWidth() << " not supported!" << endl;
				break;
			}
			shared_ptr<T> resultCPU = makeArray<T>(image.getNumPixels());
			checkCudaErrors(cudaHostRegister(resultCPU.get(), image.getNumPixels() * sizeof(T), cudaHostRegisterPortable));
			checkCudaErrors(cudaMemcpy2D(resultCPU.get(), image.getNumCols() * sizeof(T), deviceGrayImageOut, outputPitch, image.getNumCols() * sizeof(T), image.getNumRows(), cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaHostUnregister(resultCPU.get()));
			results.push_back(resultCPU);
		}
		cudaFree(deviceGrayImageIn);
		cudaFree(deviceGrayImageOut);
	}

}

