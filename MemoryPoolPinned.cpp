#include "MemoryPoolPinned.h"

#include "device_launch_parameters.h"

#include <cuda_runtime.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

namespace processing
{
	MemoryPoolPinned::~MemoryPoolPinned()
	{
		//lock_guard<mutex> lock(mutex_);
		while (buffer_.size())
		{
			checkCudaErrors(cudaFreeHost(buffer_.top()));
			buffer_.pop();
		}
	}
	shared_ptr<float> MemoryPoolPinned::acquireMemory()
	{
		//unique_lock<mutex> lock(mutex_);
		//while (buffer_.empty())
		//{
			//conditionVariable_.wait(lock);
		//}
		float* out = buffer_.top();
		buffer_.pop();
		return shared_ptr<float>(out, [this](float * ptr) { this->releaseMemory(ptr); });
	}

	void MemoryPoolPinned::releaseMemory(float* memory)
	{
		//mutex_.lock();
		buffer_.push(memory);
		//mutex_.unlock();
		//conditionVariable_.notify_one();
	}

	MemoryPoolPinned & MemoryPoolPinned::getMemoryPoolPinnedForOutput()
	{
		static MemoryPoolPinned pool(PINNED_MEMORY_BUFFER_SIZE_OUTPUT);
		return pool;
	}

	MemoryPoolPinned & MemoryPoolPinned::getMemoryPoolPinnedForInput()
	{
		static MemoryPoolPinned pool(PINNED_MEMORY_BUFFER_SIZE_INPUT);
		return pool;
	}

	MemoryPoolPinned::MemoryPoolPinned(uint bufferSize)
	{
		float* memory = nullptr;
		for (uint i = 0; i < bufferSize; i++)
		{
			checkCudaErrors(cudaMallocHost((void **)&memory, MAX_IMAGE_RESOLUTION * sizeof(float)));
			buffer_.push(memory);
		}
	};

}
