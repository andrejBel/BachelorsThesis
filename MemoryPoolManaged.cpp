#include "MemoryPoolManaged.h"

#include "device_launch_parameters.h"

#include <cuda_runtime.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include "processing.h"

namespace processing
{
	MemoryPoolManaged::~MemoryPoolManaged()
	{
		//lock_guard<mutex> lock(mutex_);
		while (buffer_.size())
		{
			checkCudaErrors(cudaFree(buffer_.top()));
			buffer_.pop();
		}
	}

	void MemoryPoolManaged::releaseMemory(float* memory)
	{
		buffer_.push(memory);
	}

	MemoryPoolManaged & MemoryPoolManaged::getMemoryPoolManaged()
	{
		static MemoryPoolManaged pool(20);
		return pool;
	}

	shared_ptr<float> MemoryPoolManaged::acquireMemory(const size_t size)
	{
		if (buffer_.size() && size <= MAX_IMAGE_RESOLUTION)
		{
			float* out = buffer_.top();
			buffer_.pop();
			return shared_ptr<float>(out, [this](float * ptr) { this->releaseMemory(ptr); });
		}
		else
		{
			float* memory = nullptr;
			checkCudaErrors(cudaMallocManaged((void **)&memory, size * sizeof(float)));
			return shared_ptr<float>(memory, [this](float * ptr) { this->releaseMemory(ptr); });
		}
	}


	MemoryPoolManaged::MemoryPoolManaged(uint bufferSize)
	{
		float* memory = nullptr;
		for (uint i = 0; i < bufferSize; i++)
		{
			checkCudaErrors(cudaMallocManaged((void **)&memory, MAX_IMAGE_RESOLUTION * sizeof(float)));
			buffer_.push(memory);
		}
	};

}
