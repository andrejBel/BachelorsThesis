#include "MemoryPoolPinned.h"

#include "device_launch_parameters.h"

#include <cuda_runtime.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include "processing.h"

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

	void MemoryPoolPinned::releaseMemory(float* memory)
	{
		//mutex_.lock();
		buffer_.push(memory);
		//mutex_.unlock();
		//conditionVariable_.notify_one();
	}

	MemoryPoolPinned & MemoryPoolPinned::getMemoryPoolPinnedForInput()
	{
		static MemoryPoolPinned pool(PINNED_MEMORY_BUFFER_SIZE_INPUT);
		return pool;
	}

	shared_ptr<float> MemoryPoolPinned::acquireMemory(const size_t size,const bool preferPinned)
	{
		if (buffer_.size() && size <= MAX_IMAGE_RESOLUTION)
		{
			float* out = buffer_.top();
			buffer_.pop();
			return shared_ptr<float>(out, [this](float * ptr) { this->releaseMemory(ptr); });
		} else
		{
			if (preferPinned) // cuda host
			{
				return allocateCudaHostSafe<float>(size);
			}
			else 
			{
				return shared_ptr<float>(new float[size], [](float * ptr) {delete[] ptr; });
			}	
		}
	}

	MemoryPoolPinned & MemoryPoolPinned::getMemoryPoolPinnedForOutput()
	{
		static MemoryPoolPinned pool(PINNED_MEMORY_BUFFER_SIZE_OUTPUT);
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
