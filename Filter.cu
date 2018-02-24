#include "Filter.h"

#include <cuda_runtime.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <processing.h>
#include <algorithm>

namespace processing 
{
	template<typename T, int WIDTH>
	Filter<T, WIDTH>::Filter(vector<T> filter, const T multiplier)
	{
		std::copy(filter.data(), filter.data() + WIDTH*WIDTH, filter_);
		if (multiplier != 1.0) 
		{
			for (uint i = 0; i < WIDTH*WIDTH; ++i)
			{
				filter_[i] *= multiplier;
			}
		}
	}

	template<typename T, int WIDTH>
	Filter<T, WIDTH>::Filter(T * filter, const T multiplier)
	{
		std::copy(filter, filter + WIDTH*WIDTH, filter_);
		if (multiplier != 1.0)
		{
			for (uint i = 0; i < WIDTH*WIDTH; ++i)
			{
				filter_[i] *= multiplier;
			}
		}
	}

	template<typename T, int WIDTH>
	CPU void Filter<T, WIDTH>::copyWholeFilterToDeviceMemory(void * destination) const
	{
		checkCudaErrors(cudaMemcpy(destination, this, sizeof(Filter<T, WIDTH>), cudaMemcpyHostToDevice));
	}



}
