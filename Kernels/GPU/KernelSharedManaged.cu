#include "KernelSharedManaged.h"


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
#include <algorithm>
#include <type_traits>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <stack>
#include <queue>
#include <atomic>
#include "MemoryPoolPinned.h"
#include "MemoryPoolPitched.h"
#include <map>
#include <utility>
#include <MemoryPoolManaged.h>

using namespace std;


#define IMAD(a, b, c) ( ((a) * (b)) + (c) )
#define CEIL(a, b) ((a + b - 1) / b)
#define ROUNDUP(a, b) (((a + b - 1) / b) * b)


#define CONVOLUTIONSHAREDTHREADSSMALL(FILTERWIDTH, BLOCKSIZEX, BLOCKSIZEY, TILESIZEX, TILESIZEY) \
		case FILTERWIDTH: \
		{ \
			cudaMemcpyToSymbolAsync(FILTERCUDA, job.filters_.get(), sizeof(float) * FILTERWIDTH * FILTERWIDTH * job.filterCount_, 0, cudaMemcpyHostToDevice, stream.stream_); \
			const short MAX_SMALL_TILE_DIMENION_X = 2; \
			const short MAX_SMALL_TILE_DIMENION_Y = 2; \
			int colsForGridX = CEIL(job.numCols, MAX_SMALL_TILE_DIMENION_X); \
			int rowsForGridY = CEIL(job.numRows, MAX_SMALL_TILE_DIMENION_Y); \
			const int FILTER_WIDTH = FILTERWIDTH; \
			const int BLOCK_SIZE_X = BLOCKSIZEX; \
			const int BLOCK_SIZE_Y = BLOCKSIZEY; \
			const int TILE_SIZE_X = TILESIZEX; \
			const int TILE_SIZE_Y = TILESIZEY; \
			const dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y); \
			const dim3 gridSize((colsForGridX + TILE_SIZE_X - 1) / TILE_SIZE_X, (rowsForGridY + TILE_SIZE_Y - 1) / TILE_SIZE_Y, 1); \
			switch (job.filterCount_)  \
			{ \
			case 1: convolutionGPUSharedThreadsSmall< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, 1> << <gridSize, blockSize, 0, stream.stream_ >> > (job.inputImage_, job.numCols, job.numRows, pitchInput_ / sizeof(float)); break; \
			case 2: convolutionGPUSharedThreadsSmall< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, 2> << <gridSize, blockSize, 0, stream.stream_ >> > (job.inputImage_, job.numCols, job.numRows, pitchInput_ / sizeof(float)); break; \
			case 3: convolutionGPUSharedThreadsSmall< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, 3> << <gridSize, blockSize, 0, stream.stream_ >> > (job.inputImage_, job.numCols, job.numRows, pitchInput_ / sizeof(float)); break; \
			case 4: convolutionGPUSharedThreadsSmall< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, 4> << <gridSize, blockSize, 0, stream.stream_ >> > (job.inputImage_, job.numCols, job.numRows, pitchInput_ / sizeof(float)); break; \
			case 5: convolutionGPUSharedThreadsSmall< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, 5> << <gridSize, blockSize, 0, stream.stream_ >> > (job.inputImage_, job.numCols, job.numRows, pitchInput_ / sizeof(float)); break; \
			case 6: convolutionGPUSharedThreadsSmall< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, 6> << <gridSize, blockSize, 0, stream.stream_ >> > (job.inputImage_, job.numCols, job.numRows, pitchInput_ / sizeof(float)); break; \
			case 7: convolutionGPUSharedThreadsSmall< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, 7> << <gridSize, blockSize, 0, stream.stream_ >> > (job.inputImage_, job.numCols, job.numRows, pitchInput_ / sizeof(float)); break; \
			case 8: convolutionGPUSharedThreadsSmall< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, 8> << <gridSize, blockSize, 0, stream.stream_ >> > (job.inputImage_, job.numCols, job.numRows, pitchInput_ / sizeof(float)); break; \
			case 9: convolutionGPUSharedThreadsSmall< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, 9> << <gridSize, blockSize, 0, stream.stream_ >> > (job.inputImage_, job.numCols, job.numRows, pitchInput_ / sizeof(float)); break; \
			case 10: convolutionGPUSharedThreadsSmall< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, 10> << <gridSize, blockSize, 0, stream.stream_ >> > (job.inputImage_, job.numCols, job.numRows, pitchInput_ / sizeof(float)); break; \
			} \
			break;  \
		} 

#define CONVOLUTIONSHAREDTHREADSBIG(FILTERWIDTH, BLOCKSIZEX, BLOCKSIZEY, TILESIZEX, TILESIZEY) \
		case FILTERWIDTH: \
		{ \
			cudaMemcpyToSymbolAsync(FILTERCUDA, job.filters_.get(), sizeof(float) * FILTERWIDTH * FILTERWIDTH * job.filterCount_, 0, cudaMemcpyHostToDevice, stream.stream_); \
			const short MAX_SMALL_TILE_DIMENION_X = 3; \
			const short MAX_SMALL_TILE_DIMENION_Y = 3; \
			int colsForGridX = CEIL(job.numCols, MAX_SMALL_TILE_DIMENION_X); \
			int rowsForGridY = CEIL(job.numRows, MAX_SMALL_TILE_DIMENION_Y); \
			const int FILTER_WIDTH = FILTERWIDTH; \
			const int BLOCK_SIZE_X = BLOCKSIZEX; \
			const int BLOCK_SIZE_Y = BLOCKSIZEY; \
			const int TILE_SIZE_X = TILESIZEX; \
			const int TILE_SIZE_Y = TILESIZEY; \
			const dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y); \
			const dim3 gridSize((colsForGridX + TILE_SIZE_X - 1) / TILE_SIZE_X, (rowsForGridY + TILE_SIZE_Y - 1) / TILE_SIZE_Y, 1); \
			switch (job.filterCount_)  \
			{ \
			case 1: convolutionGPUSharedThreadsBig< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, 1> << <gridSize, blockSize, 0, stream.stream_ >> > (job.inputImage_, job.numCols, job.numRows, pitchInput_ / sizeof(float)); break; \
			case 2: convolutionGPUSharedThreadsBig< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, 2> << <gridSize, blockSize, 0, stream.stream_ >> > (job.inputImage_, job.numCols, job.numRows, pitchInput_ / sizeof(float)); break; \
			case 3: convolutionGPUSharedThreadsBig< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, 3> << <gridSize, blockSize, 0, stream.stream_ >> > (job.inputImage_, job.numCols, job.numRows, pitchInput_ / sizeof(float)); break; \
			} \
			break;  \
		} 

#define CONVOLUTIONSHAREDTHREADFULL(FILTERWIDTH, BLOCKSIZEX, BLOCKSIZEY) \
		case FILTERWIDTH: \
		{ \
			cudaMemcpyToSymbolAsync(FILTERCUDA, job.filters_.get(), sizeof(float) * FILTERWIDTH * FILTERWIDTH * job.filterCount_, 0, cudaMemcpyHostToDevice, stream.stream_); \
			const short MAX_SMALL_TILE_DIMENION_X = 3; \
			const short MAX_SMALL_TILE_DIMENION_Y = 3; \
			int colsForGridX = CEIL(job.numCols, MAX_SMALL_TILE_DIMENION_X); \
			int rowsForGridY = CEIL(job.numRows, MAX_SMALL_TILE_DIMENION_Y); \
			const int FILTER_WIDTH = FILTERWIDTH; \
			const int BLOCK_SIZE_X = BLOCKSIZEX; \
			const int BLOCK_SIZE_Y = BLOCKSIZEY; \
			const dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y); \
			const dim3 gridSize((colsForGridX + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (rowsForGridY + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y, 1); \
			switch (job.filterCount_)  \
			{ \
			case 1: convolutionGPUSharedFullBlock< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, 1,  MAX_SMALL_TILE_DIMENION_X, MAX_SMALL_TILE_DIMENION_Y> << <gridSize, blockSize, 0, stream.stream_ >> > (job.inputImage_, job.numCols, job.numRows, pitchInput_ / sizeof(float)); break; \
			case 2: convolutionGPUSharedFullBlock< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, 2,  MAX_SMALL_TILE_DIMENION_X, MAX_SMALL_TILE_DIMENION_Y> << <gridSize, blockSize, 0, stream.stream_ >> > (job.inputImage_, job.numCols, job.numRows, pitchInput_ / sizeof(float)); break; \
			} \
			break;  \
		} 



namespace processing
{
	namespace NSKernelSharedManaged {



		namespace static_if_detail {

			struct identity {
				template<typename T>
				__device__ T operator()(T&& x) const {
					return std::forward<T>(x);
				}
			};

			template<bool Cond>
			struct statement {
				template<typename F>
				__device__ void then(const F& f) {
					f(identity());
				}

				template<typename F>
				__device__ void else_(const F&) {}
			};

			template<>
			struct statement<false> {
				template<typename F>
				__device__ void then(const F&) {}

				template<typename F>
				__device__ void else_(const F& f) {
					f(identity());
				}
			};

		} 



		template<bool Cond, typename F>
		__device__ static_if_detail::statement<Cond> static_if(F const& f) {
			static_if_detail::statement<Cond> if_;
			if_.then(f);
			return if_;
		}

		template< typename int N, typename int M>
		__device__ constexpr bool smaller()
		{
			return N < M;
		}

		template< typename int N, typename int M>
		__device__ constexpr bool bigger()
		{
			return N > M;
		}

		template< typename int N, typename int M>
		__device__ constexpr bool equal()
		{
			return N == M;
		}

		template<typename int FILTER_WIDTH, typename int FILTER_COUNT, typename int PLUSX, typename int PLUSY, typename int SHARED, typename int SMALLTILEX = PLUSX + 1, typename int SMALLTILEY = PLUSY + 1>
		__device__ __forceinline__ bool count(int2 absoluteImagePosition, int imageWidth, int imageHeight,int threadX,int threadY, float shared[][SHARED])
		{
			if (absoluteImagePosition.x + PLUSX < imageWidth  && absoluteImagePosition.y + PLUSY < imageHeight)
			{
				float results[SMALLTILEY][SMALLTILEX];
				float filterValue = 0.0;
				float * outputImage;
#pragma unroll FILTER_COUNT
				for (int i = 0; i < FILTER_COUNT; ++i)
				{
					outputImage = MANAGEDOUTPUT[i];
#pragma unroll SMALLTILEY 
					for (int y = 0; y < SMALLTILEY; y++)
					{
#pragma unroll SMALLTILEX
						for (int x = 0; x < SMALLTILEX; x++)
						{
							results[y][x] = 0;
						}
					}

#pragma unroll FILTER_WIDTH
					for (int yOffset = 0; yOffset < FILTER_WIDTH; yOffset++)
					{
#pragma unroll FILTER_WIDTH
						for (int xOffset = 0; xOffset < FILTER_WIDTH; xOffset++)
						{
							filterValue = FILTERCUDA[yOffset*FILTER_WIDTH + xOffset + FILTER_WIDTH * FILTER_WIDTH * i];
#pragma unroll SMALLTILEY 
							for (int y = 0; y < SMALLTILEY; y++)
							{
#pragma unroll SMALLTILEX
								for (int x = 0; x < SMALLTILEX; x++)
								{
									results[y][x] += filterValue * shared[yOffset + threadY + y][xOffset + threadX + x];
								}
							}
						}
					}

#pragma unroll SMALLTILEY 
					for (int y = 0; y < SMALLTILEY; y++)
					{
						static_if<equal<SMALLTILEX, 1>()>([&absoluteImagePosition, imageWidth, outputImage, y = y, results = results](auto f) {
							*((float *)(outputImage + IMAD(absoluteImagePosition.y + y, imageWidth, absoluteImagePosition.x))) = *(float *)(&results[y]);

						});
						static_if<equal<SMALLTILEX, 2>()>([&absoluteImagePosition, imageWidth, outputImage, y = y, results = results](auto f) {
							*((float2 *)(outputImage + IMAD(absoluteImagePosition.y + y, imageWidth, absoluteImagePosition.x))) = *(float2 *)(&results[y]);
						});
						static_if<equal<SMALLTILEX, 3>()>([&absoluteImagePosition, imageWidth, outputImage, y = y, results = results](auto f) {
							*((float3 *)(outputImage + IMAD(absoluteImagePosition.y + y, imageWidth, absoluteImagePosition.x))) = *(float3 *)(&results[y]);
						});
					}
					static_if<bigger<FILTER_COUNT, 1>()>([](auto f) {
						__syncthreads();
					});
				}
				return true;
			}
			return false;
		}
		


		template<typename int FILTER_WIDTH, typename int BLOCK_SIZE_X, typename int BLOCK_SIZE_Y, typename int FILTER_COUNT, typename int TILE_SIZE_X, typename int TILE_SIZE_Y>
		__global__ void convolutionGPUSharedFullBlock(const float * __restrict__ inputImage, int imageWidth, int imageHeight, int inputPitch)
		{
			int2 absoluteImagePosition;
			absoluteImagePosition.x = IMAD(blockIdx.x, BLOCK_SIZE_X, threadIdx.x) * TILE_SIZE_X;
			absoluteImagePosition.y = IMAD(blockIdx.y, BLOCK_SIZE_Y, threadIdx.y) * TILE_SIZE_Y;

			int2 positionShared;
			positionShared.x = blockIdx.x * BLOCK_SIZE_X * TILE_SIZE_X;
			positionShared.y = blockIdx.y * BLOCK_SIZE_Y * TILE_SIZE_Y;
			__shared__ float shared[BLOCK_SIZE_Y*TILE_SIZE_Y + (FILTER_WIDTH - 1) + TILE_SIZE_Y][BLOCK_SIZE_X *TILE_SIZE_X + (FILTER_WIDTH - 1) + TILE_SIZE_X];
			int threadX = threadIdx.x * TILE_SIZE_X;
			int threadY = threadIdx.y * TILE_SIZE_Y;
			float3 row;
			for (int j = threadY; j < FILTER_WIDTH - 1 + BLOCK_SIZE_Y * TILE_SIZE_Y; j += BLOCK_SIZE_Y * TILE_SIZE_Y)
			{
				for (int i = threadX; i < FILTER_WIDTH - 1 + BLOCK_SIZE_X * TILE_SIZE_X; i += BLOCK_SIZE_X * TILE_SIZE_X)
				{
#pragma unroll TILE_SIZE_Y
					for (int k = 0; k < TILE_SIZE_Y; k++)
					{
						row = *(float3 *)(inputImage + IMAD(positionShared.y + j + k, inputPitch, positionShared.x + i));
						shared[j + k][i] = row.x;
						shared[j + k][i + 1] = row.y;
						shared[j + k][i + 2] = row.z;
					}

				}
			}
			__syncthreads();

			imageWidth -= (FILTER_WIDTH - 1);
			imageHeight -= (FILTER_WIDTH - 1);

			
			bool done = count<FILTER_WIDTH, FILTER_COUNT, 2, 2 >(absoluteImagePosition, imageWidth, imageHeight, threadX, threadY, shared);
			if (!done)
			{
				done = count<FILTER_WIDTH, FILTER_COUNT, 2, 1 >(absoluteImagePosition, imageWidth, imageHeight, threadX, threadY, shared);
				if (!done)
				{
					done = count<FILTER_WIDTH, FILTER_COUNT, 1, 2 >(absoluteImagePosition, imageWidth, imageHeight, threadX, threadY, shared);
					if (!done)
					{
						done = count<FILTER_WIDTH, FILTER_COUNT, 0, 2 >(absoluteImagePosition, imageWidth, imageHeight, threadX, threadY, shared);
						if (!done)
						{
							done = count<FILTER_WIDTH, FILTER_COUNT, 2, 0 >(absoluteImagePosition, imageWidth, imageHeight, threadX, threadY, shared);
							if (!done)
							{
								done = count<FILTER_WIDTH, FILTER_COUNT, 1, 1 >(absoluteImagePosition, imageWidth, imageHeight, threadX, threadY, shared);
								if (!done)
								{
									done = count<FILTER_WIDTH, FILTER_COUNT, 1, 0 >(absoluteImagePosition, imageWidth, imageHeight, threadX, threadY, shared);
									if (!done)
									{
										done = count<FILTER_WIDTH, FILTER_COUNT, 0, 1 >(absoluteImagePosition, imageWidth, imageHeight, threadX, threadY, shared);
										if (!done)
										{
											done = count<FILTER_WIDTH, FILTER_COUNT, 0, 0 >(absoluteImagePosition, imageWidth, imageHeight, threadX, threadY, shared);
										}
									}
								}
							}
						}
					}
				}
			}
			

			

			
		}

		template<typename int FILTER_WIDTH, typename int BLOCK_SIZE_X, typename int BLOCK_SIZE_Y, typename int TILE_SIZE_X, typename int TILE_SIZE_Y, typename int FILTER_COUNT>
		__global__ void convolutionGPUSharedThreadsBig(const float * __restrict__ inputImage, int imageWidth, int imageHeight, int inputPitch)
		{
			const int smallTile = 3;
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
				imageWidth -= (FILTER_WIDTH - 1);
				imageHeight -= (FILTER_WIDTH - 1);

				
				bool done = count<FILTER_WIDTH, FILTER_COUNT, 2, 2 >(absoluteImagePosition, imageWidth, imageHeight, threadX, threadY, shared);
				if (!done)
				{
					done = count<FILTER_WIDTH, FILTER_COUNT, 2, 1 >(absoluteImagePosition, imageWidth, imageHeight, threadX, threadY, shared);
					if (!done)
					{
						done = count<FILTER_WIDTH, FILTER_COUNT, 1, 2 >(absoluteImagePosition, imageWidth, imageHeight, threadX, threadY, shared);
						if (!done)
						{
							done = count<FILTER_WIDTH, FILTER_COUNT, 0, 2 >(absoluteImagePosition, imageWidth, imageHeight, threadX, threadY, shared);
							if (!done)
							{
								done = count<FILTER_WIDTH, FILTER_COUNT, 2, 0 >(absoluteImagePosition, imageWidth, imageHeight, threadX, threadY, shared);
								if (!done)
								{
									done = count<FILTER_WIDTH, FILTER_COUNT, 1, 1 >(absoluteImagePosition, imageWidth, imageHeight, threadX, threadY, shared);
									if (!done)
									{
										done = count<FILTER_WIDTH, FILTER_COUNT, 1, 0 >(absoluteImagePosition, imageWidth, imageHeight, threadX, threadY, shared);
										if (!done)
										{
											done = count<FILTER_WIDTH, FILTER_COUNT, 0, 1 >(absoluteImagePosition, imageWidth, imageHeight, threadX, threadY, shared);
											if (!done)
											{
												done = count<FILTER_WIDTH, FILTER_COUNT, 0, 0 >(absoluteImagePosition, imageWidth, imageHeight, threadX, threadY, shared);
											}
										}
									}
								}
							}
						}
					}
				}
				
				

			}
		}


		template<typename int FILTER_WIDTH, typename int BLOCK_SIZE_X, typename int BLOCK_SIZE_Y, typename int TILE_SIZE_X, typename int TILE_SIZE_Y, typename int FILTER_COUNT>
		__global__ void convolutionGPUSharedThreadsSmall(const float * __restrict__ inputImage, int imageWidth, int imageHeight, int inputPitch)
		{
			constexpr int smallTile = 2;
			__shared__ float shared[BLOCK_SIZE_Y * smallTile][BLOCK_SIZE_X * smallTile];
			const int threadX = threadIdx.x * smallTile;
			const int threadY = threadIdx.y * smallTile;
			int2 absoluteImagePosition;
			absoluteImagePosition.x = IMAD(blockIdx.x, TILE_SIZE_X, threadIdx.x) * smallTile;
			absoluteImagePosition.y = IMAD(blockIdx.y, TILE_SIZE_Y, threadIdx.y) * smallTile;
			float2 row = *(float2 *)(inputImage + IMAD(absoluteImagePosition.y, inputPitch, absoluteImagePosition.x));
			//float2 secondRow = *(float2 *)(inputImage + IMAD(absoluteImagePosition.y + 1, imageWidth, absoluteImagePosition.x));
			shared[threadY][threadX] = row.x;
			shared[threadY][threadX + 1] = row.y;
			row = *(float2 *)(inputImage + IMAD(absoluteImagePosition.y + 1, inputPitch, absoluteImagePosition.x));
			shared[threadY + 1][threadX] = row.x;
			shared[threadY + 1][threadX + 1] = row.y;
			__syncthreads();
			if (threadX < TILE_SIZE_X * smallTile && threadY < TILE_SIZE_Y * smallTile)
			{

				imageWidth -= (FILTER_WIDTH - 1);
				imageHeight -= (FILTER_WIDTH - 1);

				bool done = count<FILTER_WIDTH, FILTER_COUNT, 1, 1 >(absoluteImagePosition, imageWidth, imageHeight,threadX,threadY , shared);
				if (!done) 
				{
					done = count<FILTER_WIDTH, FILTER_COUNT, 1, 0 >(absoluteImagePosition, imageWidth, imageHeight, threadX, threadY, shared);
					if (!done) 
					{
						done = count<FILTER_WIDTH, FILTER_COUNT, 0, 1 >(absoluteImagePosition, imageWidth, imageHeight, threadX, threadY, shared);
						if (!done) 
						{
							done = count<FILTER_WIDTH, FILTER_COUNT, 0, 0 >(absoluteImagePosition, imageWidth, imageHeight, threadX, threadY, shared);
						}
					}
				}
			}
		}


		static constexpr int MAX_JOB_SIZE = 10;
		static const vector<int> jobLimits =
		{
			0, //0
			MAX_JOB_SIZE,//1
			0,//2
			MAX_JOB_SIZE,//3
			0,//4
			MAX_JOB_SIZE,//5
			0,//6
			4,//7
			0,//8
			3,////9
			0,//10
			2, ////11
			0,//12
			1,//13
			0,//14
			1//15
		};




		CudaStream streams[3];
		std::queue<Job> jobs_;
		std::stack<float *> inputImages_;
		queue<Job> jobsInPostProcess_;

		bool preprocessPrepared_ = false;

		bool processFinished = false;

		mutex mutexInputImages_;
		mutex mutexJobs_;
		mutex mutexProcessPostProcess_;

		condition_variable conditionVariable_;
		size_t pitchInput_;


		void preprocess(CudaStream& stream, vector<shared_ptr<ImageFactory>>& images, vector<shared_ptr<Filter>>& filters)
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
			pitchInput_ = MemoryPoolPitched::getMemoryPoolPitchedForInput().getPitch();
			vector<float *> inputImagesMemory = MemoryPoolPitched::getMemoryPoolPitchedForInput().getMemory();
			for (float * memory : inputImagesMemory)
			{
				inputImages_.push(memory);
			}

			map<int, vector<shared_ptr<Filter>>> filterSets;
			for (shared_ptr<Filter>& filter : filters)
			{
				filterSets[filter->getWidth()].push_back(filter);
			}
			vector<FilterBox> filtersHostMemories;

			for (std::pair< int, vector<shared_ptr<Filter>> > filters : filterSets)
			{
				int filterWidth = filters.first;
				int filterCount = filters.second.size();
				int sizeOfFilter = filters.first * filters.first;
				const int jobSize = std::min(jobLimits.at(filterWidth), 10);
				for (int i = 0; i < filterCount; i += jobSize)
				{
					int startCopiing = i;
					int endCopiing = std::min(i + jobSize - 1, filterCount - 1);
					int filterJobSize = endCopiing - startCopiing + 1;
					shared_ptr<float> memoryForFilters = shared_ptr<float>(new float[sizeOfFilter * filterJobSize], [](float * ptr) { delete[] ptr; });
					for (int offset = 0, index = startCopiing; index <= endCopiing; ++index, offset += sizeOfFilter)
					{
						std::copy(filters.second[index]->getFilter(), filters.second[index]->getFilter() + sizeOfFilter, memoryForFilters.get() + offset);
					}
					FilterBox box;
					box.filterCount_ = filterJobSize;
					box.filterWidth_ = filterWidth;
					box.memory_ = memoryForFilters;
					filtersHostMemories.push_back(box);
				}
			}

			for (vector<shared_ptr<ImageFactory>>::size_type i = 0; i < images.size(); i++)
			{
				shared_ptr<ImageFactory>& image = images[i];
				int numberOfColumns = image->getNumCols();
				int numberOfRows = image->getNumRows();
				float * hostGrayImage = image->getInputGrayPointerFloat();
				unique_lock<mutex> lock(mutexInputImages_);
				while (inputImages_.empty())
				{
					conditionVariable_.wait(lock);
				}
				float * deviceGrayImageIn = inputImages_.top();
				inputImages_.pop();
				lock.unlock();
				checkCudaErrors(cudaMemcpy2DAsync(deviceGrayImageIn, pitchInput_, hostGrayImage, numberOfColumns * sizeof(float), numberOfColumns * sizeof(float), numberOfRows, cudaMemcpyHostToDevice, stream.stream_));
				checkCudaErrors(cudaStreamSynchronize(stream.stream_));
				vector<Job> jobs;
				for (FilterBox filters : filtersHostMemories)
				{
					Job job;
					job.filterCount_ = filters.filterCount_;
					job.filters_ = filters.memory_;
					job.filterWidth_ = filters.filterWidth_;
					job.inputImage_ = deviceGrayImageIn;
					job.numCols = numberOfColumns;
					job.numRows = numberOfRows;
					jobs.push_back(job);
				}

				mutexJobs_.lock();
				for (size_t j = 0; j < jobs.size(); j++)
				{
					if (j == jobs.size() - 1)
					{
						jobs[j].returnInputImage_ = true;
						if (i == images.size() - 1)
						{
							jobs[j].finish_ = true;
						}
					}
					jobs_.push(jobs[j]);
				}
				preprocessPrepared_ = true;
				mutexJobs_.unlock();
				conditionVariable_.notify_all();
			}
		}

		void process(CudaStream& stream, vector<shared_ptr<float>>& results)
		{
			queue<Job> jobs;
			bool end = false;
			vector<shared_ptr<float>> managedOutput(MAX_JOB_SIZE);
			vector<float* > managedOutputRow(MAX_JOB_SIZE);
			while (end == false)
			{
				unique_lock<mutex> lock(mutexJobs_);
				while (preprocessPrepared_ == false)
				{
					conditionVariable_.wait(lock);
				}
				if (jobs.size() != 0)
				{
					cout << "Mistake, jobs are not 0!!!!!" << endl;
				}
				std::swap(jobs, jobs_);
				if (jobs.size() == 0)
				{
					cout << "Mistake jobs connot be 0!!!!!" << endl;
				}
				preprocessPrepared_ = false;
				lock.unlock();
				while (jobs.size())
				{
					Job job = std::move(jobs.front());
					jobs.pop();

					for (int i = 0; i < job.filterCount_; i++)
					{
						managedOutput[i] = MemoryPoolManaged::getMemoryPoolManaged().acquireMemory(job.numCols * job.numRows);
						managedOutputRow[i] = managedOutput[i].get();
					}
					checkCudaErrors(cudaMemcpyToSymbolAsync(MANAGEDOUTPUT, managedOutputRow.data(), sizeof(float**) * job.filterCount_, 0, cudaMemcpyHostToDevice, stream.stream_));
					switch (job.filterWidth_)
					{
						CONVOLUTIONSHAREDTHREADSSMALL(1, 32, 16, 32, 16)
						CONVOLUTIONSHAREDTHREADSSMALL(3, 32, 16, 31, 15);
						CONVOLUTIONSHAREDTHREADSSMALL(5, 32, 16, 30, 14)
						CONVOLUTIONSHAREDTHREADSSMALL(7, 32, 32, 29, 29)
						CONVOLUTIONSHAREDTHREADSBIG(9, 32, 16, 29, 13)
						CONVOLUTIONSHAREDTHREADSBIG(11, 32, 16, 28, 12)
						CONVOLUTIONSHAREDTHREADFULL(13, 32, 6)
						CONVOLUTIONSHAREDTHREADFULL(15, 32, 8)
						
						
					default:
						std::cerr << "Filter with width: " << job.filterWidth_ << " not supported!" << endl;
						break;
					}
					checkCudaErrors(cudaStreamSynchronize(stream.stream_));
					if (job.returnInputImage_)
					{
						mutexInputImages_.lock();
						inputImages_.push(job.inputImage_);
						mutexInputImages_.unlock();
						conditionVariable_.notify_all();
					}
					if (job.finish_)
					{
						end = true;
					}
					for (int i = 0; i < job.filterCount_; i++)
					{
						results.push_back(managedOutput[i]);
					}

				}
			}
		}


	}

	KernelSharedManaged::KernelSharedManaged() : SimpleRunnable(true)
	{}

	void KernelSharedManaged::run(vector<shared_ptr<ImageFactory>>& images, vector<shared_ptr<Filter>>& filters, vector<shared_ptr<float>>& results)
	{
		using namespace NSKernelSharedManaged;
		thread threadPreprocessing(preprocess, std::ref(streams[0]), std::ref(images), std::ref(filters));
		thread threadProcessing(process, std::ref(streams[1]), std::ref(results));


		threadPreprocessing.join();
		threadProcessing.join();
	}

}


