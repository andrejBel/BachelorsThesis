#include "MemoryPoolPitched.h"

#include "device_launch_parameters.h"

#include <cuda_runtime.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#include <iostream>
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

	MemoryPoolPitched::MemoryPoolPitched(uint memorySize) : memory_(memorySize)
	{
		float * memory = nullptr;
		for (uint i = 0; i < memorySize; i++)
		{
			checkCudaErrors(cudaMallocPitch<float>(&memory, &pitch_, (MAX_IMAGE_WIDTH + 300) * sizeof(float), MAX_IMAGE_HEIGHT + 300));
			memory_[i] = memory;
		}
	}

	MemoryPoolPitched::~MemoryPoolPitched()
	{
		for (float * memory : memory_ )
		{
			cudaFree(memory);
		}
	}




}
