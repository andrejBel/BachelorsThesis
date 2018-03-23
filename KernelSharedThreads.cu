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

using namespace std;


#define IMAD(a, b, c) ( __mul24((a), (b)) + (c) )
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
			job.bufferStart_ =  PITCHED_MEMORY_BUFFER_HOST.acquire(job.filterCount_); \
			switch (job.filterCount_)  \
			{ \
			case 1: convolutionGPUSharedThreadsSmall< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, 1, PITCHED_MEMORY_BUFFER_SIZE_OUTPUT> << <gridSize, blockSize, 0, stream.stream_ >> > (job.numRows, job.numCols, job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float), job.bufferStart_); break; \
			case 2: convolutionGPUSharedThreadsSmall< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, 2, PITCHED_MEMORY_BUFFER_SIZE_OUTPUT> << <gridSize, blockSize, 0, stream.stream_ >> > (job.numRows, job.numCols, job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float), job.bufferStart_); break; \
			case 3: convolutionGPUSharedThreadsSmall< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, 3, PITCHED_MEMORY_BUFFER_SIZE_OUTPUT> << <gridSize, blockSize, 0, stream.stream_ >> > (job.numRows, job.numCols, job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float), job.bufferStart_); break; \
			case 4: convolutionGPUSharedThreadsSmall< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, 4, PITCHED_MEMORY_BUFFER_SIZE_OUTPUT> << <gridSize, blockSize, 0, stream.stream_ >> > (job.numRows, job.numCols, job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float), job.bufferStart_); break; \
			case 5: convolutionGPUSharedThreadsSmall< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, 5, PITCHED_MEMORY_BUFFER_SIZE_OUTPUT> << <gridSize, blockSize, 0, stream.stream_ >> > (job.numRows, job.numCols, job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float), job.bufferStart_); break; \
			case 6: convolutionGPUSharedThreadsSmall< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, 6, PITCHED_MEMORY_BUFFER_SIZE_OUTPUT> << <gridSize, blockSize, 0, stream.stream_ >> > (job.numRows, job.numCols, job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float), job.bufferStart_); break; \
			case 7: convolutionGPUSharedThreadsSmall< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, 7, PITCHED_MEMORY_BUFFER_SIZE_OUTPUT> << <gridSize, blockSize, 0, stream.stream_ >> > (job.numRows, job.numCols, job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float), job.bufferStart_); break; \
			case 8: convolutionGPUSharedThreadsSmall< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, 8, PITCHED_MEMORY_BUFFER_SIZE_OUTPUT> << <gridSize, blockSize, 0, stream.stream_ >> > (job.numRows, job.numCols, job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float), job.bufferStart_); break; \
			case 9: convolutionGPUSharedThreadsSmall< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, 9, PITCHED_MEMORY_BUFFER_SIZE_OUTPUT> << <gridSize, blockSize, 0, stream.stream_ >> > (job.numRows, job.numCols, job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float), job.bufferStart_); break; \
			case 10: convolutionGPUSharedThreadsSmall< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, 10, PITCHED_MEMORY_BUFFER_SIZE_OUTPUT> << <gridSize, blockSize, 0, stream.stream_ >> > (job.numRows, job.numCols, job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float), job.bufferStart_); break; \
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
			job.bufferStart_ =  PITCHED_MEMORY_BUFFER_HOST.acquire(job.filterCount_); \
			switch (job.filterCount_)  \
			{ \
			case 1: convolutionGPUSharedThreadsBig< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, 1, PITCHED_MEMORY_BUFFER_SIZE_OUTPUT> << <gridSize, blockSize, 0, stream.stream_ >> > (job.numRows, job.numCols, job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float), job.bufferStart_); break; \
			case 2: convolutionGPUSharedThreadsBig< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, 2, PITCHED_MEMORY_BUFFER_SIZE_OUTPUT> << <gridSize, blockSize, 0, stream.stream_ >> > (job.numRows, job.numCols, job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float), job.bufferStart_); break; \
			case 3: convolutionGPUSharedThreadsBig< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, 3, PITCHED_MEMORY_BUFFER_SIZE_OUTPUT> << <gridSize, blockSize, 0, stream.stream_ >> > (job.numRows, job.numCols, job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float), job.bufferStart_); break; \
			case 4: convolutionGPUSharedThreadsBig< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, 4, PITCHED_MEMORY_BUFFER_SIZE_OUTPUT> << <gridSize, blockSize, 0, stream.stream_ >> > (job.numRows, job.numCols, job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float), job.bufferStart_); break; \
			case 5: convolutionGPUSharedThreadsBig< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, 5, PITCHED_MEMORY_BUFFER_SIZE_OUTPUT> << <gridSize, blockSize, 0, stream.stream_ >> > (job.numRows, job.numCols, job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float), job.bufferStart_); break; \
			case 6: convolutionGPUSharedThreadsBig< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, 6, PITCHED_MEMORY_BUFFER_SIZE_OUTPUT> << <gridSize, blockSize, 0, stream.stream_ >> > (job.numRows, job.numCols, job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float), job.bufferStart_); break; \
			case 7: convolutionGPUSharedThreadsBig< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, 7, PITCHED_MEMORY_BUFFER_SIZE_OUTPUT> << <gridSize, blockSize, 0, stream.stream_ >> > (job.numRows, job.numCols, job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float), job.bufferStart_); break; \
			case 8: convolutionGPUSharedThreadsBig< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, 8, PITCHED_MEMORY_BUFFER_SIZE_OUTPUT> << <gridSize, blockSize, 0, stream.stream_ >> > (job.numRows, job.numCols, job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float), job.bufferStart_); break; \
			case 9: convolutionGPUSharedThreadsBig< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, 9, PITCHED_MEMORY_BUFFER_SIZE_OUTPUT> << <gridSize, blockSize, 0, stream.stream_ >> > (job.numRows, job.numCols, job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float), job.bufferStart_); break; \
			case 10: convolutionGPUSharedThreadsBig< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, 10, PITCHED_MEMORY_BUFFER_SIZE_OUTPUT> << <gridSize, blockSize, 0, stream.stream_ >> > (job.numRows, job.numCols, job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float), job.bufferStart_); break; \
			} \
			break;  \
		} 


namespace processing
{
	
	template<typename int FILTER_WIDTH, typename int BLOCK_SIZE_X, typename int BLOCK_SIZE_Y, typename int TILE_SIZE_X, typename int TILE_SIZE_Y, typename int FILTER_COUNT, typename int BUFFER_SIZE>
	__global__ void convolutionGPUSharedThreadsBig(const int numRows, const int numCols, const float * __restrict__ inputImage,const int inputPitch,const int outputPitch,const int bufferStartPosition)
	{
		const int smallTile = 3;
		float results[smallTile][smallTile] = { 0.0 };

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
			float * outputImage;
			float filterValue;
#pragma unroll FILTER_COUNT
			for (int i = 0; i < FILTER_COUNT; ++i)
			{
				outputImage = PITCHED_MEMORY_BUFFER_DEVICE.memory_[(bufferStartPosition + i) % BUFFER_SIZE];
				results[0][0] = 0.0;
				results[0][1] = 0.0;
				results[0][2] = 0.0;
				results[1][0] = 0.0;
				results[1][1] = 0.0;
				results[1][2] = 0.0;
				results[2][0] = 0.0;
				results[2][1] = 0.0;
				results[2][2] = 0.0;

				
#pragma unroll FILTER_WIDTH
				for (int yOffset = 0; yOffset < FILTER_WIDTH; yOffset++)
				{
#pragma unroll FILTER_WIDTH
					for (int xOffset = 0; xOffset < FILTER_WIDTH; xOffset++)
					{
						filterValue = FILTERCUDA[yOffset*FILTER_WIDTH + xOffset + FILTER_WIDTH * FILTER_WIDTH * i];
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
				*(float3 *)(outputImage + IMAD(absoluteImagePosition.y , outputPitch, absoluteImagePosition.x)) = *(float3 *)(&results[0]);
				*(float3 *)(outputImage + IMAD(absoluteImagePosition.y + 1, outputPitch, absoluteImagePosition.x)) = *(float3 *)(&results[1]);
				*(float3 *)(outputImage + IMAD(absoluteImagePosition.y + 2, outputPitch, absoluteImagePosition.x)) = *(float3 *)(&results[2]);
				__syncthreads();
			}
		}
	}


	template<typename int FILTER_WIDTH, typename int BLOCK_SIZE_X, typename int BLOCK_SIZE_Y, typename int TILE_SIZE_X, typename int TILE_SIZE_Y, typename int FILTER_COUNT, typename int BUFFER_SIZE>
	__global__ void convolutionGPUSharedThreadsSmall(const int numRows, const int numCols, const float * __restrict__ inputImage, const int inputPitch, const int outputPitch, const int bufferStartPosition)
	{
		const int smallTile = 2;
		__shared__ float shared[BLOCK_SIZE_Y * smallTile][BLOCK_SIZE_X * smallTile];
		const int threadX = threadIdx.x * smallTile;
		const int threadY = threadIdx.y * smallTile;
		int2 absoluteImagePosition;
		absoluteImagePosition.x = IMAD(blockIdx.x, TILE_SIZE_X, threadIdx.x) * smallTile;
		absoluteImagePosition.y = IMAD(blockIdx.y, TILE_SIZE_Y, threadIdx.y) * smallTile;
		float2 row = *(float2 *)(inputImage + IMAD(absoluteImagePosition.y, inputPitch, absoluteImagePosition.x));
		//float2 secondRow = *(float2 *)(inputImage + IMAD(absoluteImagePosition.y + 1, inputPitch, absoluteImagePosition.x));
		shared[threadY][threadX] = row.x;
		shared[threadY][threadX + 1] = row.y;
		row = *(float2 *)(inputImage + IMAD(absoluteImagePosition.y + 1, inputPitch, absoluteImagePosition.x));
		shared[threadY + 1][threadX] = row.x;
		shared[threadY + 1][threadX + 1] = row.y;
		__syncthreads();
		if (threadX < TILE_SIZE_X * smallTile && threadY < TILE_SIZE_Y * smallTile)
		{
			float2 resultRow1;
			float2 resultRow2;
			float filterValue = 0.0;
			float * outputImage;
#pragma unroll FILTER_COUNT
			for (int i = 0; i < FILTER_COUNT; ++i)
			{
				outputImage = PITCHED_MEMORY_BUFFER_DEVICE.memory_[(bufferStartPosition + i) % BUFFER_SIZE];
				resultRow1.x = 0.0;
				resultRow1.y = 0.0;
				resultRow2.x = 0.0;
				resultRow2.y = 0.0;
#pragma unroll FILTER_WIDTH
				for (int yOffset = 0; yOffset < FILTER_WIDTH; yOffset++)
				{
#pragma unroll FILTER_WIDTH
					for (int xOffset = 0; xOffset < FILTER_WIDTH; xOffset++)
					{
						filterValue = FILTERCUDA[yOffset*FILTER_WIDTH + xOffset + FILTER_WIDTH * FILTER_WIDTH * i];
						resultRow1.x += filterValue * shared[yOffset + threadY][xOffset + threadX];
						resultRow1.y += filterValue * shared[yOffset + threadY][xOffset + threadX + 1];
						resultRow2.x += filterValue * shared[yOffset + threadY + 1][xOffset + threadX];
						resultRow2.y += filterValue * shared[yOffset + threadY + 1][xOffset + threadX + 1];
					}
				}
				*((float2 *)(outputImage + IMAD(absoluteImagePosition.y, outputPitch, absoluteImagePosition.x))) = resultRow1;
				*((float2 *)(outputImage + IMAD(absoluteImagePosition.y + 1, outputPitch, absoluteImagePosition.x))) = resultRow2;
				__syncthreads();
			}
		}
	}

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
	size_t pitchOutput_;

	void preprocess(CudaStream& stream, vector<shared_ptr<ImageFactory>>& images, vector<shared_ptr<AbstractFilter>>& filters)
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
		pitchInput_ = MemoryPoolPitched::getMemoryPoolPitchedForInput().getPitch();
		pitchOutput_ = MemoryPoolPitched::getMemoryPoolPitchedForOutput().getPitch();
		map<int, vector<shared_ptr<AbstractFilter>>> filterSets;
		for (shared_ptr<AbstractFilter>& filter : filters)
		{
			filterSets[filter->getWidth()].push_back(filter);
		}
		vector<FilterBox> filtersHostMemories;

		for (std::pair< int, vector<shared_ptr<AbstractFilter>> > filters : filterSets)
		{
			int filterWidth = filters.first;
			int filterCount = filters.second.size();
			int sizeOfFilter = filters.first * filters.first;
			for (int i = 0; i < filterCount; i += PITCHED_MEMORY_BUFFER_SIZE_OUTPUT)
			{
				int startCopiing = i;
				int endCopiing = std::min(i + PITCHED_MEMORY_BUFFER_SIZE_OUTPUT - 1, filterCount - 1);
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
			checkCudaErrors(cudaStreamSynchronize(stream.stream_));
			preprocessPrepared_ = true;
			mutexJobs_.unlock();
			conditionVariable_.notify_all();
		}
	}

	void process(CudaStream& stream)
	{
		queue<Job> jobs;
		bool end = false;
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
				switch (job.filterWidth_)
				{
					//CONVOLUTIONSHAREDTHREADSSMALL(1, 32, 16, 32, 16)
					//CONVOLUTIONSHAREDTHREADSSMALL(3, 32, 16, 31, 15)
					CONVOLUTIONSHAREDTHREADSSMALL(5, 32, 16, 30, 14)
					//CONVOLUTIONSHAREDTHREADSSMALL(7, 32, 32, 29, 29)
					//CONVOLUTIONSHAREDTHREADSSMALL(9, 32, 32, 28, 28)
					//CONVOLUTIONSHAREDTHREADSSMALL(11, 32, 32, 27, 27)
					//CONVOLUTIONSHAREDTHREADSSMALL(13, 32, 32, 26, 26)
					CONVOLUTIONSHAREDTHREADSBIG(15, 32, 18, 27, 13)
				default:
					std::cerr << "Filter with width: " << job.filterWidth_ << " not supported!" << endl;
					break;
				}
				checkCudaErrors(cudaStreamSynchronize(stream.stream_));
				lock.lock();
				jobsInPostProcess_.push(job);
				lock.unlock();
				conditionVariable_.notify_all();
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

	void postprocess(CudaStream& stream, vector<shared_ptr<float>>& results)
	{
		bool end = false;
		Job job;
		queue<Job> jobs;
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

				for (int i = 0; i < job.filterCount_; i++)
				{
					shared_ptr<float> resultCPU = MemoryPoolPinned::getMemoryPoolPinnedForOutput().acquireMemory();
					checkCudaErrors(cudaMemcpy2DAsync(resultCPU.get(), xlen * sizeof(float), PITCHED_MEMORY_BUFFER_HOST.memory_[(job.bufferStart_ + i) % PITCHED_MEMORY_BUFFER_SIZE_OUTPUT], pitchOutput_, xlen * sizeof(float), ylen, cudaMemcpyDeviceToHost, stream.stream_));
					checkCudaErrors(cudaStreamSynchronize(stream.stream_));
					results.push_back(resultCPU);
				}
				PITCHED_MEMORY_BUFFER_HOST.release(job.filterCount_);
				if (job.finish_)
				{
					end = true;
				}
			}
		}
	}






	KernelSharedThreads::KernelSharedThreads()
	{}


	void KernelSharedThreads::run(ImageFactory& image, vector<shared_ptr<AbstractFilter>>& filters, vector<shared_ptr<float>>& results)
	{

		vector<shared_ptr<ImageFactory>> images;
		images.push_back(shared_ptr<ImageFactory>(&image, [](ImageFactory * ptr) {}));
		images.push_back(shared_ptr<ImageFactory>(&image, [](ImageFactory * ptr) {}));
		thread threadPreprocessing(preprocess, std::ref(streams[0]), std::ref(images), std::ref(filters));
		thread threadProcessing(process, std::ref(streams[1]));
		thread threadPostprocessing(postprocess, std::ref(streams[2]), std::ref(results));

		threadPreprocessing.join();
		threadProcessing.join();
		threadPostprocessing.join();


	}

}


