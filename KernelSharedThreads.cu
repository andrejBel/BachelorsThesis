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


#define CONVOLUTIONSHAREDTHREADS(FILTERWIDTH, BLOCKSIZEX, BLOCKSIZEY, TILESIZEX, TILESIZEY) \
		case FILTERWIDTH: \
		{ \
			cudaMemcpyToSymbolAsync(FILTERCUDA, job.filters_.get(), sizeof(float) * FILTERWIDTH * FILTERWIDTH * job.filterCount_, 0, cudaMemcpyHostToDevice, stream.stream_); \
			const int FILTER_WIDTH = FILTERWIDTH; \
			const int BLOCK_SIZE_X = BLOCKSIZEX; \
			const int BLOCK_SIZE_Y = BLOCKSIZEY; \
			const int TILE_SIZE_X = TILESIZEX; \
			const int TILE_SIZE_Y = TILESIZEY; \
			const dim3 blockSize(BLOCK_SIZE_X, BLOCK_SIZE_Y); \
			const dim3 gridSize((colsForGridX + TILE_SIZE_X - 1) / TILE_SIZE_X, (rowsForGridY + TILE_SIZE_Y - 1) / TILE_SIZE_Y, 1); \
			unique_lock<mutex> lock(mutexProcessPostProcess_); \
			while (postprocessFinished_ == false) \
			{ \
				conditionVariable_.wait(lock); \
			} \
			postprocessFinished_ = false; \
			lock.unlock(); \
			switch (job.filterCount_)  \
			{ \
			case 1: convolutionGPUSharedThreads< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, 1> << <gridSize, blockSize, 0, stream.stream_ >> > (job.numRows, job.numCols, job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float)); break; \
			case 2: convolutionGPUSharedThreads< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, 2> << <gridSize, blockSize, 0, stream.stream_ >> > (job.numRows, job.numCols, job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float)); break; \
			case 3: convolutionGPUSharedThreads< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, 3> << <gridSize, blockSize, 0, stream.stream_ >> > (job.numRows, job.numCols, job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float)); break; \
			case 4: convolutionGPUSharedThreads< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, 4> << <gridSize, blockSize, 0, stream.stream_ >> > (job.numRows, job.numCols, job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float)); break; \
			case 5: convolutionGPUSharedThreads< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, 5> << <gridSize, blockSize, 0, stream.stream_ >> > (job.numRows, job.numCols, job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float)); break; \
			case 6: convolutionGPUSharedThreads< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, 6> << <gridSize, blockSize, 0, stream.stream_ >> > (job.numRows, job.numCols, job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float)); break; \
			case 7: convolutionGPUSharedThreads< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, 7> << <gridSize, blockSize, 0, stream.stream_ >> > (job.numRows, job.numCols, job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float)); break; \
			case 8: convolutionGPUSharedThreads< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, 8> << <gridSize, blockSize, 0, stream.stream_ >> > (job.numRows, job.numCols, job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float)); break; \
			case 9: convolutionGPUSharedThreads< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, 9> << <gridSize, blockSize, 0, stream.stream_ >> > (job.numRows, job.numCols, job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float)); break; \
			case 10: convolutionGPUSharedThreads< FILTER_WIDTH, BLOCK_SIZE_X, BLOCK_SIZE_Y, TILE_SIZE_X, TILE_SIZE_Y, 10> << <gridSize, blockSize, 0, stream.stream_ >> > (job.numRows, job.numCols, job.inputImage_, pitchInput_ / sizeof(float), pitchOutput_ / sizeof(float)); break; \
			} \
			break;  \
		} 




namespace processing
{
	template<typename int FILTER_WIDTH, typename int BLOCK_SIZE_X, typename int BLOCK_SIZE_Y, typename int TILE_SIZE_X, typename int TILE_SIZE_Y, typename int FILTER_COUNT>
	__global__ void convolutionGPUSharedThreads(const int numRows, const int numCols, const float * __restrict__ inputImage, int inputPitch, int outputPitch)
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
			float result1 = 0.0; //00
			float result2 = 0.0; //01
			float result3 = 0.0; //10
			float result4 = 0.0; //11
			float filterValue = 0.0;
			float * outputImage;
#pragma unroll FILTER_COUNT
			for (int i = 0; i < FILTER_COUNT; ++i)
			{
				outputImage = PITCHED_MEMORY_BUFFER_DEVICE.memory_[i];
#pragma unroll FILTER_WIDTH
				for (int yOffset = 0; yOffset < FILTER_WIDTH; yOffset++)
				{
#pragma unroll FILTER_WIDTH
					for (int xOffset = 0; xOffset < FILTER_WIDTH; xOffset++)
					{
						filterValue = FILTERCUDA[yOffset*FILTER_WIDTH + xOffset + FILTER_WIDTH * FILTER_WIDTH * i];
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
				result1 = 0.0;
				result2 = 0.0; 
				result3 = 0.0;
				result4 = 0.0; 
				__syncthreads();
			}
		}
	}

	CudaStream streams[3];
	std::queue<Job> jobs_;
	std::stack<float *> inputImages_;
	Job jobInPostProcess_;

	bool preprocessPrepared_ = false;

	bool processFinished = false;
	bool postprocessFinished_ = true;

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
		const short MAX_SMALL_TILE_DIMENION_X = 2;
		const short MAX_SMALL_TILE_DIMENION_Y = 2;
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

				int colsForGridX = CEIL(job.numCols, MAX_SMALL_TILE_DIMENION_X);
				int rowsForGridY = CEIL(job.numRows, MAX_SMALL_TILE_DIMENION_Y);

				switch (job.filterWidth_)
				{

					CONVOLUTIONSHAREDTHREADS(1, 32, 16, 32, 16)
					CONVOLUTIONSHAREDTHREADS(3, 32, 16, 31, 15)
					CONVOLUTIONSHAREDTHREADS(5, 32, 16, 30, 14)
					CONVOLUTIONSHAREDTHREADS(7, 32, 32, 29, 29)
					CONVOLUTIONSHAREDTHREADS(9, 32, 32, 28, 28)
					CONVOLUTIONSHAREDTHREADS(11, 32, 32, 27, 27)
					CONVOLUTIONSHAREDTHREADS(13, 32, 32, 26, 26)
					CONVOLUTIONSHAREDTHREADS(15, 32, 16, 25, 9)
				default:
					std::cerr << "Filter with width: " << job.filterWidth_ << " not supported!" << endl;
					break;
				}
				jobInPostProcess_ = job;
				checkCudaErrors(cudaStreamSynchronize(stream.stream_));
				lock.lock();
				processFinished = true;
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
		while (end == false)
		{
			unique_lock<mutex> lock(mutexProcessPostProcess_);
			while (processFinished == false)
			{
				conditionVariable_.wait(lock);
			}
			job = jobInPostProcess_;
			processFinished = false;
			lock.unlock();
			int xlen = job.numCols - (job.filterWidth_ - 1);
			int ylen = job.numRows - (job.filterWidth_ - 1);

			for (int i = 0; i < job.filterCount_; i++)
			{
				shared_ptr<float> resultCPU = MemoryPoolPinned::getMemoryPoolPinnedForOutput().acquireMemory();
				checkCudaErrors(cudaMemcpy2DAsync(resultCPU.get(), xlen * sizeof(float), PITCHED_MEMORY_BUFFER_HOST.memory_[i], pitchOutput_, xlen * sizeof(float), ylen, cudaMemcpyDeviceToHost, stream.stream_));
				checkCudaErrors(cudaStreamSynchronize(stream.stream_));
				results.push_back(resultCPU);
			}


			lock.lock();
			postprocessFinished_ = true;
			lock.unlock();
			conditionVariable_.notify_all();
			if (job.finish_)
			{
				end = true;
			}
		}
	}






	KernelSharedThreads::KernelSharedThreads()
	{}


	void KernelSharedThreads::run(ImageFactory& image, vector<shared_ptr<AbstractFilter>>& filters, vector<shared_ptr<float>>& results)
	{

		vector<shared_ptr<ImageFactory>> images;
		images.push_back(shared_ptr<ImageFactory>(&image, [](ImageFactory * ptr) {}));
		//images.push_back(shared_ptr<ImageFactory>(&image, [](ImageFactory * ptr) {}));
		thread threadPreprocessing(preprocess, std::ref(streams[0]), std::ref(images), std::ref(filters));
		thread threadProcessing(process, std::ref(streams[1]));
		thread threadPostprocessing(postprocess, std::ref(streams[2]), std::ref(results));

		threadPreprocessing.join();
		threadProcessing.join();
		threadPostprocessing.join();


	}

}


