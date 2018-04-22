#include "MemoryPoolPitched.h"

#include "device_launch_parameters.h"

#include <cuda_runtime.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <algorithm>
namespace processing 
{
	

	MemoryPoolPitched & MemoryPoolPitched::getMemoryPoolPitchedForOutput()
	{
		static MemoryPoolPitched pool(PITCHED_MEMORY_BUFFER_SIZE_OUTPUT);
		return pool;
	}

	MemoryPoolPitched & MemoryPoolPitched::getMemoryPoolPitchedForInput()
	{
		static MemoryPoolPitched pool(PITCHED_MEMORY_BUFFER_SIZE_INPUT);
		return pool;
	}

	void MemoryPoolPitched::realoc(const int maxImageWidth, const int maxImageHeight)
	{
		if (maxImageWidth > maxImageWidth_ || maxImageHeight >  maxImageHeight_) 
		{
			maxImageWidth_ = std::max(maxImageWidth_, maxImageWidth);
			maxImageHeight_ = std::max(maxImageHeight_, maxImageHeight);
			float * memory = nullptr;
			for (uint i = 0; i < memory_.size(); i++)
			{
				if (memory_[i])
				{
					checkCudaErrors(cudaFree(memory_[i]));
				}
				checkCudaErrors(cudaMallocPitch<float>(&memory, &pitch_, (maxImageWidth_ + IMAGE_ADDITIONAL_PIXELS) * sizeof(float), maxImageHeight_ + IMAGE_ADDITIONAL_PIXELS));
				memory_[i] = memory;
			}
		}
		
	}

	MemoryPoolPitched::MemoryPoolPitched(uint memorySize) : 
		memory_(memorySize, nullptr),
		maxImageWidth_(MAX_IMAGE_WIDTH ),
		maxImageHeight_(MAX_IMAGE_HEIGHT)
	{
		float * memory = nullptr;
		for (uint i = 0; i < memorySize; i++)
		{
			checkCudaErrors(cudaMallocPitch<float>(&memory, &pitch_, (maxImageWidth_ + IMAGE_ADDITIONAL_PIXELS) * sizeof(float), maxImageHeight_ + IMAGE_ADDITIONAL_PIXELS));
			memory_[i] = memory;
		}
	}

	MemoryPoolPitched::~MemoryPoolPitched()
	{
		for (float * memory : memory_ )
		{
			checkCudaErrors(cudaFree(memory));
		}
	}




}
