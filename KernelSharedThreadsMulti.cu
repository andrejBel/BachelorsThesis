#include "KernelSharedThreadsMulti.h"
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

#include <algorithm>
#include <type_traits>
#include <utility>
#include "MemoryPoolPinned.h"
#include "MemoryPoolPitched.h"
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <stack>
#include <queue>
#include <map>
#include <utility>
#include "ThreadPool.h"

#define MUL(a, b) __mul24(a, b)
#define IMAD(a, b, c) ( __mul24((a), (b)) + (c) )
#define CEIL(a, b) ((a + b - 1) / b)
#define ROUNDUP(a, b) (((a + b - 1) / b) * b)

using namespace std;
namespace processing
{

#define CONVOLUTIONSHAREDMULTIFULLBLOCK(FILTERWIDTH, BLOCKSIZEX, BLOCKSIZEY) \
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
							const dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y); \
							const dim3 gridSize((colsForGridX + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X, (rowsForGridY + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y, 1); \
							if (job.makeZeros) \
							{ \
								switch (job.filterCount_) \
								{ \
								case 1: convolutionGPUSharedFullBlock< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, 1, PITCHED_MEMORY_BUFFER_SIZE_OUTPUT, MAX_SMALL_TILE_DIMENION_X, MAX_SMALL_TILE_DIMENION_Y, true> << <gridSize, blockSize, 0, stream.stream_ >> > (job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float), job.bufferStart_); break; \
								case 2: convolutionGPUSharedFullBlock< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, 2, PITCHED_MEMORY_BUFFER_SIZE_OUTPUT, MAX_SMALL_TILE_DIMENION_X, MAX_SMALL_TILE_DIMENION_Y, true> << <gridSize, blockSize, 0, stream.stream_ >> > (job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float), job.bufferStart_); break; \
								case 3: convolutionGPUSharedFullBlock< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, 3, PITCHED_MEMORY_BUFFER_SIZE_OUTPUT, MAX_SMALL_TILE_DIMENION_X, MAX_SMALL_TILE_DIMENION_Y, true> << <gridSize, blockSize, 0, stream.stream_ >> > (job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float), job.bufferStart_); break; \
								case 4: convolutionGPUSharedFullBlock< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, 4, PITCHED_MEMORY_BUFFER_SIZE_OUTPUT, MAX_SMALL_TILE_DIMENION_X, MAX_SMALL_TILE_DIMENION_Y, true> << <gridSize, blockSize, 0, stream.stream_ >> > (job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float), job.bufferStart_); break; \
								case 5: convolutionGPUSharedFullBlock< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, 5, PITCHED_MEMORY_BUFFER_SIZE_OUTPUT, MAX_SMALL_TILE_DIMENION_X, MAX_SMALL_TILE_DIMENION_Y, true> << <gridSize, blockSize, 0, stream.stream_ >> > (job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float), job.bufferStart_); break; \
								case 6: convolutionGPUSharedFullBlock< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, 6, PITCHED_MEMORY_BUFFER_SIZE_OUTPUT, MAX_SMALL_TILE_DIMENION_X, MAX_SMALL_TILE_DIMENION_Y, true> << <gridSize, blockSize, 0, stream.stream_ >> > (job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float), job.bufferStart_); break; \
								case 7: convolutionGPUSharedFullBlock< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, 7, PITCHED_MEMORY_BUFFER_SIZE_OUTPUT, MAX_SMALL_TILE_DIMENION_X, MAX_SMALL_TILE_DIMENION_Y, true> << <gridSize, blockSize, 0, stream.stream_ >> > (job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float), job.bufferStart_); break; \
								case 8: convolutionGPUSharedFullBlock< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, 8, PITCHED_MEMORY_BUFFER_SIZE_OUTPUT, MAX_SMALL_TILE_DIMENION_X, MAX_SMALL_TILE_DIMENION_Y, true> << <gridSize, blockSize, 0, stream.stream_ >> > (job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float), job.bufferStart_); break; \
								case 9: convolutionGPUSharedFullBlock< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, 9, PITCHED_MEMORY_BUFFER_SIZE_OUTPUT, MAX_SMALL_TILE_DIMENION_X, MAX_SMALL_TILE_DIMENION_Y, true> << <gridSize, blockSize, 0, stream.stream_ >> > (job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float), job.bufferStart_); break; \
								case 10: convolutionGPUSharedFullBlock< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, 10, PITCHED_MEMORY_BUFFER_SIZE_OUTPUT, MAX_SMALL_TILE_DIMENION_X, MAX_SMALL_TILE_DIMENION_Y, true> << <gridSize, blockSize, 0, stream.stream_ >> > (job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float), job.bufferStart_); break; \
								} \
							} \
							else  \
							{ \
								switch (job.filterCount_) \
								{ \
								case 1: convolutionGPUSharedFullBlock< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, 1, PITCHED_MEMORY_BUFFER_SIZE_OUTPUT, MAX_SMALL_TILE_DIMENION_X, MAX_SMALL_TILE_DIMENION_Y, false> << <gridSize, blockSize, 0, stream.stream_ >> > (job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float), job.bufferStart_); break; \
								case 2: convolutionGPUSharedFullBlock< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, 2, PITCHED_MEMORY_BUFFER_SIZE_OUTPUT, MAX_SMALL_TILE_DIMENION_X, MAX_SMALL_TILE_DIMENION_Y, false> << <gridSize, blockSize, 0, stream.stream_ >> > (job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float), job.bufferStart_); break; \
								case 3: convolutionGPUSharedFullBlock< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, 3, PITCHED_MEMORY_BUFFER_SIZE_OUTPUT, MAX_SMALL_TILE_DIMENION_X, MAX_SMALL_TILE_DIMENION_Y, false> << <gridSize, blockSize, 0, stream.stream_ >> > (job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float), job.bufferStart_); break; \
								case 4: convolutionGPUSharedFullBlock< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, 4, PITCHED_MEMORY_BUFFER_SIZE_OUTPUT, MAX_SMALL_TILE_DIMENION_X, MAX_SMALL_TILE_DIMENION_Y, false> << <gridSize, blockSize, 0, stream.stream_ >> > (job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float), job.bufferStart_); break;  \
								case 5: convolutionGPUSharedFullBlock< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, 5, PITCHED_MEMORY_BUFFER_SIZE_OUTPUT, MAX_SMALL_TILE_DIMENION_X, MAX_SMALL_TILE_DIMENION_Y, false> << <gridSize, blockSize, 0, stream.stream_ >> > (job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float), job.bufferStart_); break; \
								case 6: convolutionGPUSharedFullBlock< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, 6, PITCHED_MEMORY_BUFFER_SIZE_OUTPUT, MAX_SMALL_TILE_DIMENION_X, MAX_SMALL_TILE_DIMENION_Y, false> << <gridSize, blockSize, 0, stream.stream_ >> > (job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float), job.bufferStart_); break; \
								case 7: convolutionGPUSharedFullBlock< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, 7, PITCHED_MEMORY_BUFFER_SIZE_OUTPUT, MAX_SMALL_TILE_DIMENION_X, MAX_SMALL_TILE_DIMENION_Y, false> << <gridSize, blockSize, 0, stream.stream_ >> > (job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float), job.bufferStart_); break; \
								case 8: convolutionGPUSharedFullBlock< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, 8, PITCHED_MEMORY_BUFFER_SIZE_OUTPUT, MAX_SMALL_TILE_DIMENION_X, MAX_SMALL_TILE_DIMENION_Y, false> << <gridSize, blockSize, 0, stream.stream_ >> > (job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float), job.bufferStart_); break; \
								case 9: convolutionGPUSharedFullBlock< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, 9, PITCHED_MEMORY_BUFFER_SIZE_OUTPUT, MAX_SMALL_TILE_DIMENION_X, MAX_SMALL_TILE_DIMENION_Y, false> << <gridSize, blockSize, 0, stream.stream_ >> > (job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float), job.bufferStart_); break; \
								case 10: convolutionGPUSharedFullBlock< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, 10, PITCHED_MEMORY_BUFFER_SIZE_OUTPUT, MAX_SMALL_TILE_DIMENION_X, MAX_SMALL_TILE_DIMENION_Y, false> << <gridSize, blockSize, 0, stream.stream_ >> > (job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float), job.bufferStart_); break; \
								} \
							} \
							break; \
						}

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

	} //end of namespace static_if_detail



	template<bool Cond, typename F>
	__device__ static_if_detail::statement<Cond> static_if(F const& f) {
		static_if_detail::statement<Cond> if_;
		if_.then(f);
		return if_;
	}

	template<typename int FILTER_WIDTH, typename int BLOCK_SIZE_X, typename int BLOCK_SIZE_Y, typename int TILE_SIZE_X, typename int TILE_SIZE_Y, typename int MAX_SMALL_TILE_DIMENION_X = 2, typename int MAX_SMALL_TILE_DIMENION_Y = 2>
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
			*((float2 *)&shared[threadY + i][threadX]) = *(float2 *)(inputImage + IMAD(absoluteImagePosition.y + i, inputPitch, absoluteImagePosition.x));
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


	template<typename int FILTER_WIDTH, typename int BLOCK_SIZE_X, typename int BLOCK_SIZE_Y, typename int FILTER_COUNT, typename int BUFFER_SIZE, typename int TILE_SIZE_X, typename int TILE_SIZE_Y, typename bool MAKEZEROS>
	__global__ void convolutionGPUSharedFullBlock(const float * __restrict__ inputImage, const int inputPitch, const int outputPitch, const short bufferStartPosition)
	{
		float results[TILE_SIZE_X * TILE_SIZE_Y];
		int2 absoluteImagePosition;
		absoluteImagePosition.x = (blockIdx.x *  BLOCK_SIZE_X + threadIdx.x) * TILE_SIZE_X;
		absoluteImagePosition.y = (blockIdx.y * BLOCK_SIZE_Y + threadIdx.y) * TILE_SIZE_Y;
		int2 positionShared;
		positionShared.x = blockIdx.x * BLOCK_SIZE_X * TILE_SIZE_X;
		positionShared.y = blockIdx.y * BLOCK_SIZE_Y * TILE_SIZE_Y;
		__shared__ float shared[BLOCK_SIZE_Y * TILE_SIZE_Y + (FILTER_WIDTH - 1)][BLOCK_SIZE_X *TILE_SIZE_X + (FILTER_WIDTH - 1)];
		int threadX = threadIdx.x * TILE_SIZE_X;
		int threadY = threadIdx.y * TILE_SIZE_Y;
		for (int j = threadY; j < FILTER_WIDTH - 1 + BLOCK_SIZE_Y * TILE_SIZE_Y; j += BLOCK_SIZE_Y * TILE_SIZE_Y)
		{
			for (int i = threadX; i < FILTER_WIDTH - 1 + BLOCK_SIZE_X * TILE_SIZE_X; i += BLOCK_SIZE_X * TILE_SIZE_X)
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
		float * outputImage;
		float filterValue;

#pragma unroll FILTER_COUNT
		for (int i = 0; i < FILTER_COUNT; ++i)
		{
			outputImage = PITCHED_MEMORY_BUFFER_DEVICE.memory_[(bufferStartPosition + i) % BUFFER_SIZE];
#pragma unroll TILE_SIZE_Y
			for (int k = 0; k < TILE_SIZE_Y; k++)
			{
#pragma unroll TILE_SIZE_X
				for (int l = 0; l < TILE_SIZE_X; l++)
				{
					results[k * TILE_SIZE_Y + l] = 0.0;
				}
			}
#pragma unroll FILTER_WIDTH
			for (int yOffset = 0; yOffset < FILTER_WIDTH; yOffset++)
			{
#pragma unroll FILTER_WIDTH
				for (int xOffset = 0; xOffset < FILTER_WIDTH; xOffset++)
				{
					filterValue = FILTERCUDA[yOffset*FILTER_WIDTH + xOffset + FILTER_WIDTH * FILTER_WIDTH * i];
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
					
					static_if<MAKEZEROS>([outputImage, absoluteImagePosition, outputPitch, &results,k,l, TILE_SIZE_Y = TILE_SIZE_Y](auto f) {
						outputImage[IMAD(absoluteImagePosition.y + k, outputPitch, absoluteImagePosition.x + l)] = results[k * TILE_SIZE_Y + l];
						//static_assert(0 > 1, "false");
					}).else_([outputImage, absoluteImagePosition, outputPitch, &results, k,l, TILE_SIZE_Y = TILE_SIZE_Y](auto f) {
						outputImage[IMAD(absoluteImagePosition.y + k, outputPitch, absoluteImagePosition.x + l)] += results[k * TILE_SIZE_Y + l];
						//static_assert(0 > 1, "false") 
					});

					/*
					if (MAKEZEROS)
					{
						outputImage[IMAD(absoluteImagePosition.y + k, outputPitch, absoluteImagePosition.x + l)] = results[k * TILE_SIZE_Y + l];
					}
					else
					{
						outputImage[IMAD(absoluteImagePosition.y + k, outputPitch, absoluteImagePosition.x + l)] += results[k * TILE_SIZE_Y + l];
					}
					*/
				}
			}
			__syncthreads();
		}
	}

	namespace KernelSharedThreadsMultiNS
	{
		// variables
		CudaStream streams[3];
		std::queue<Job> jobsInProcess_;
		queue<Job> jobsInPostProcess_;
		std::stack<float *> inputImages_;
		bool preprocessPrepared_ = false;

		size_t pitchInput_;
		size_t pitchOutput_;

		mutex mutexInputImages_;
		mutex mutexJobInProcess_;
		mutex mutexProcessPostProcess_;

		condition_variable conditionVariable_;

		ThreadPool threadPool(2);

		void printJob(Job& job) 
		{
			cout << "-------------------------------------" << endl;
			cout << "Filter count: " << job.filterCount_ << endl;
			cout << "Buffer start: " << job.bufferStart_ << endl;
			cout << "Filter group start index: " << job.filterGroupStartIndex << endl;
			cout << "Filters: " << job.filters_ << endl;
			cout << "Filter width: " << job.filterWidth_ << endl;
			cout << "Finish: " << job.finish_ << endl;
			cout << "Go to postprocess: " << job.goToPostprocess << endl;
			cout << "Input image: " << job.inputImage_ << endl;
			cout << "Return image: " << job.returnInputImage_ << endl;
			cout << "Make zeros: " << job.makeZeros << endl;
			cout << "Num cols: " << job.numCols << endl;
			cout << "Num rows: " << job.numRows << endl;
			cout << "*************************************" << endl;
		}


		void preprocess(CudaStream& stream, vector<shared_ptr<ImageFactory>>& images, vector<vector<shared_ptr<Filter>>>& filters)
		{
			for (int i = 0; i < PITCHED_MEMORY_BUFFER_SIZE_OUTPUT; i++)
			{
				PITCHED_MEMORY_BUFFER_HOST.memory_[i] = MemoryPoolPitched::getMemoryPoolPitchedForOutput().getMemory()[i];
				//cout << "Index host: " << i << ", value: " << PITCHED_MEMORY_BUFFER_HOST.memory_[i] << endl;
			}
			cudaMemcpyToSymbolAsync(PITCHED_MEMORY_BUFFER_DEVICE.memory_, PITCHED_MEMORY_BUFFER_HOST.memory_, sizeof(float**) * PITCHED_MEMORY_BUFFER_SIZE_OUTPUT, 0, cudaMemcpyHostToDevice, stream.stream_);

			vector<float *> inputImagesMemory = MemoryPoolPitched::getMemoryPoolPitchedForInput().getMemory();
			
			for (float * memory : inputImagesMemory)
			{
				inputImages_.push(memory);
			}
			/*
			for (size_t i = 0; i < inputImagesMemory.size(); i++)
			{
				cout << i << ": " << inputImagesMemory[i] << endl;
			}
			*/
			pitchInput_ = MemoryPoolPitched::getMemoryPoolPitchedForInput().getPitch();
			pitchOutput_ = MemoryPoolPitched::getMemoryPoolPitchedForOutput().getPitch();

			// roztriedenie po sirkach filtra
			const int numCols = images[0]->getNumCols();
			const int numRows = images[0]->getNumRows();

			const int imageSize = images.size();
			const int groupSize = filters.size();
			constexpr int OUTPUT_SIZE = PITCHED_MEMORY_BUFFER_SIZE_OUTPUT / 2;


			vector<vector<shared_ptr<Filter>>> filtersCopy(filters);
			std::sort(filtersCopy.begin(), filtersCopy.end(), [](vector<shared_ptr<Filter>>& first, vector<shared_ptr<Filter>> second) 
			{
				return first[0]->getWidth() < second[0]->getWidth();
			});
			vector<BatchGroup> filterGroups;
			for (int filterGroupIndex = 0; filterGroupIndex < groupSize; filterGroupIndex += OUTPUT_SIZE)
			{
				int startOfGroup = filterGroupIndex;
				int endOfGroup = std::min(filterGroupIndex + OUTPUT_SIZE - 1, groupSize - 1);
				int totalGroupSize = endOfGroup - startOfGroup + 1;
				BatchGroup group;
				group.filterCount_ = totalGroupSize;
				group.filterStart_ = startOfGroup;
				group.filterEnd_ = endOfGroup;
				map<short, short> filterWidths;
				for (int k = startOfGroup; k <= endOfGroup; k++)
				{
					if (filterWidths.find(filtersCopy[k][0]->getWidth()) == filterWidths.end()) // neobsahuje 
					{
						filterWidths[filtersCopy[k][0]->getWidth()] = 1;
					}
					else // obsahuje
					{
						filterWidths[filtersCopy[k][0]->getWidth()] += 1;
					}
				}
				for (std::pair<int, int> filterSet : filterWidths)
				{
					group.filterInfos_.push_back(BatchGroupInfo(filterSet.first, filterSet.second));
				}
				filterGroups.push_back(std::move(group));
			}


			vector<float *> deviceInputImagesForFilterGroup(imageSize, nullptr);
			for (int i = 0; i < imageSize; i += OUTPUT_SIZE )
			{
				int startOfImage = i;
				int endOfImage = std::min(i + OUTPUT_SIZE - 1, imageSize - 1);
				int totalUsedImages = endOfImage - startOfImage + 1;
				
				
				for (BatchGroup& group : filterGroups)
				{
					int bufferOutputStart = PITCHED_MEMORY_BUFFER_HOST.acquire(group.filterCount_);
					for (int imageIndex = startOfImage; imageIndex <= endOfImage; ++imageIndex)
					{
					
						if (deviceInputImagesForFilterGroup[imageIndex] == nullptr) 
						{
							shared_ptr<ImageFactory>& image = images[imageIndex];
							float * hostGrayImage = image->getInputGrayPointerFloat();
							unique_lock<mutex> lock(mutexInputImages_);
							while (inputImages_.empty())
							{
								conditionVariable_.wait(lock);
							}
							float * deviceGrayImageIn = inputImages_.top();
							inputImages_.pop();
							lock.unlock();
							deviceInputImagesForFilterGroup[imageIndex] = deviceGrayImageIn;
							checkCudaErrors(cudaMemcpy2DAsync(deviceGrayImageIn, pitchInput_, hostGrayImage, numCols * sizeof(float), numCols * sizeof(float), numRows, cudaMemcpyHostToDevice, stream.stream_));
							checkCudaErrors(cudaStreamSynchronize(stream.stream_));
						}
						int filterGroupIndexStart = group.filterStart_;
						int localBufferOutputStart = bufferOutputStart;
						vector<Job> jobs;
						for (BatchGroupInfo& info : group.filterInfos_)
						{
							Job job;
							job.numCols = numCols;
							job.numRows = numRows;
							job.inputImage_ = deviceInputImagesForFilterGroup[imageIndex];
							job.filterCount_ = info.filterCount_;
							job.filterWidth_ = info.filterWidth_;
							job.bufferStart_ = localBufferOutputStart;
							localBufferOutputStart += info.filterCount_;
							
							int sizeOfFilter = job.filterWidth_* job.filterWidth_;
							shared_ptr<float> memoryForFilters = shared_ptr<float>(new float[sizeOfFilter * job.filterCount_], [](float * ptr) { delete[] ptr; });
							for (int offset = 0, index = filterGroupIndexStart; index < filterGroupIndexStart + job.filterCount_; ++index, offset += sizeOfFilter)
							{
								std::copy(filtersCopy[index][imageIndex]->getFilter(), filtersCopy[index][imageIndex]->getFilter() + sizeOfFilter, memoryForFilters.get() + offset);
							}
							job.filterGroupStartIndex = filterGroupIndexStart;
							filterGroupIndexStart += job.filterCount_;
							job.filters_ = memoryForFilters;
							

							if (imageIndex == startOfImage) // prvy prvok, nuluje buffer output
							{
								job.makeZeros = true;
							}
							if (&group == &filterGroups[filterGroups.size() - 1]) // posledna grupa, vracia vstupnz obrayok
							{
								job.returnInputImage_ = true;
							}
							if (imageIndex == endOfImage)
							{
								job.goToPostprocess = true;
							}
							jobs.push_back(std::move(job));
						}
						mutexJobInProcess_.lock();
						for (int j = 0; j < jobs.size(); j++)
						{
							if (imageIndex == imageSize - 1 && j == jobs.size() - 1 && &group == &filterGroups[filterGroups.size() - 1])
							{
								jobs[j].finish_ = true;
							}
							//printJob(jobs[j]);
							jobsInProcess_.push(std::move(jobs[j]));
						}
						
						preprocessPrepared_ = true;
						mutexJobInProcess_.unlock();
						conditionVariable_.notify_all();
					}
				}

			}

			cout << "koniec preprocessing" << endl;
		}

		void process(CudaStream& stream)
		{
			queue<Job> jobs;
			bool end = false;
			while (end == false) 
			{
				unique_lock<mutex> lock(mutexJobInProcess_);
				while (preprocessPrepared_ == false)
				{
					conditionVariable_.wait(lock);
				}
				if (jobs.size() != 0)
				{
					cout << "Mistake, jobs are not 0!!!!!" << endl;
				}
				std::swap(jobs, jobsInProcess_);
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

					switch (job.filterWidth_)
					{
						CONVOLUTIONSHAREDMULTIFULLBLOCK(1, 32, 16)
						CONVOLUTIONSHAREDMULTIFULLBLOCK(3,32,16)
						CONVOLUTIONSHAREDMULTIFULLBLOCK(5, 32, 16)
						//CONVOLUTIONSHAREDMULTIFULLBLOCK(7, 32, 16)
						//CONVOLUTIONSHAREDMULTIFULLBLOCK(9, 32, 16)
						//CONVOLUTIONSHAREDMULTIFULLBLOCK(11, 32, 16)
						//CONVOLUTIONSHAREDMULTIFULLBLOCK(13, 32, 16)
						//CONVOLUTIONSHAREDMULTIFULLBLOCK(15, 32, 16)
					default:
						std::cerr << "Filter with width: " << job.filterWidth_ << " not supported!" << endl;
						break;
					}

					checkCudaErrors(cudaStreamSynchronize(stream.stream_));
					if (job.goToPostprocess) 
					{
						mutexProcessPostProcess_.lock();
						jobsInPostProcess_.push(job);
						mutexProcessPostProcess_.unlock();
						conditionVariable_.notify_all();
					}
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
				}
			}
		}

		void postprocess(CudaStream& stream, vector<shared_ptr<float>>& results, size_t filterGroupSize, int numberOfCols, int numberOfRows)
		{
			bool end = false;
			Job job;
			queue<Job> jobs;
			results.resize(filterGroupSize);

			shared_ptr<float> bufferForCopiing[] = { MemoryPoolPinned::getMemoryPoolPinnedForOutput().acquireMemory(), MemoryPoolPinned::getMemoryPoolPinnedForOutput().acquireMemory() };
			int indexInBuffer = 0;
			const size_t pixels = numberOfCols * numberOfRows;
			for (size_t i = 0; i < filterGroupSize; i++)
			{
				//results[i] = makeArray<float>(pixels);
				results[i] = MemoryPoolPinned::getMemoryPoolPinnedForOutput().acquireMemory();
				std::fill(results[i].get(), results[i].get() + pixels, 0);
				/*
				double difference = pixels;
				difference = std::ceil(difference / threadPool.getThreadCount());
				size_t leftBorder = 0;
				size_t rightBorder = size_t(difference);
				for (uint j = 0; j < threadPool.getThreadCount(); ++j)
				{
					threadPool.addTask([leftBorder, rightBorder, result = results[i].get()]()
					{
						std::fill(result + leftBorder, result + rightBorder, 0);
					});
					leftBorder = rightBorder;
					rightBorder = std::min(rightBorder + size_t(difference), size_t(pixels));
				}
				*/
			}
			//threadPool.finishAll();
			while (end == false)
			{
				unique_lock<mutex> lock(mutexProcessPostProcess_);
				while (jobsInPostProcess_.empty())
				{
					conditionVariable_.wait(lock);
				}
				std::swap(jobs, jobsInPostProcess_);
				lock.unlock();

				while (jobs.size())
				{
					job = std::move(jobs.front());
					jobs.pop();
					
					int xlen = job.numCols - (job.filterWidth_ - 1);
					int ylen = job.numRows - (job.filterWidth_ - 1);
					const size_t pixels = xlen * ylen;
					for (int i = 0; i < job.filterCount_; i++)
					{
						
						checkCudaErrors(cudaMemcpy2DAsync(bufferForCopiing[indexInBuffer].get(), xlen * sizeof(float), PITCHED_MEMORY_BUFFER_HOST.memory_[(job.bufferStart_ + i) % PITCHED_MEMORY_BUFFER_SIZE_OUTPUT], pitchOutput_, xlen * sizeof(float), ylen, cudaMemcpyDeviceToHost, stream.stream_));
						threadPool.finishAll();
						checkCudaErrors(cudaStreamSynchronize(stream.stream_));

						PITCHED_MEMORY_BUFFER_HOST.release(1);
						int filterGroupIndex = job.filterGroupStartIndex + i;
						//std::transform(results[filterGroupIndex].get(), results[filterGroupIndex].get() + pixels, bufferForCopiing.get(), results[filterGroupIndex].get(), std::plus<float>());
						
						double difference = pixels;
						difference = std::ceil(difference / threadPool.getThreadCount());
						size_t leftBorder = 0;
						size_t rightBorder = size_t(difference);
						for (uint i = 0; i < threadPool.getThreadCount(); ++i)
						{
							threadPool.addTask([leftBorder, rightBorder, dst = results[filterGroupIndex].get(), src = bufferForCopiing[indexInBuffer].get()]()
							{
								std::transform(dst + leftBorder, dst + rightBorder, src + leftBorder, dst + leftBorder, std::plus<float>());
							});
							leftBorder = rightBorder;
							rightBorder = std::min(rightBorder + size_t(difference), size_t(pixels));
						}
						++indexInBuffer;
						indexInBuffer %= 2;
					}
					if (job.finish_)
					{
						end = true;
					}
				}
			}

		}
	}
	

	KernelSharedThreadsMulti::KernelSharedThreadsMulti()
	{
	}

	void KernelSharedThreadsMulti::run(ImageFactory & image, vector<shared_ptr<Filter>>& filters, vector<shared_ptr<float>>& results)
	{
		throw std::runtime_error("Simple convolution not supported");
	}

	void KernelSharedThreadsMulti::run(vector<shared_ptr<ImageFactory>>& images, vector<vector<shared_ptr<Filter>>>& filters, vector<shared_ptr<float>>& results)
	{
		pair<bool, string> checking = controlInputForMultiConvolution(images, filters);
		if (checking.first == false)
		{
			cerr << checking.second << endl;
			return;
		}
		using namespace KernelSharedThreadsMultiNS;
		
		thread threadProcessing(process, std::ref(streams[1]));
		thread threadPreprocessing(preprocess, std::ref(streams[0]), std::ref(images), std::ref(filters));
		thread threadPostprocessing(postprocess, std::ref(streams[2]), std::ref(results), filters.size(), images[0]->getNumCols() , images[0]->getNumRows());
		threadPreprocessing.join();
		threadProcessing.join();
		threadPostprocessing.join();
	}

}