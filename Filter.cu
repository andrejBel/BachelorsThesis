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
	template<typename T, int FILTER_WIDTH>
	Filter<T, FILTER_WIDTH>::Filter(vector<T> filter, const T multiplier)
	{
		std::copy(filter.data(), filter.data() + FILTER_WIDTH*FILTER_WIDTH, filter_);
		if (multiplier != 1.0) 
		{
			for (uint i = 0; i < FILTER_WIDTH*FILTER_WIDTH; ++i)
			{
				filter_[i] *= multiplier;
			}
		}
	}

	template<typename T, int FILTER_WIDTH>
	Filter<T, FILTER_WIDTH>::Filter(T * filter, const T multiplier)
	{
		std::copy(filter, filter + FILTER_WIDTH*FILTER_WIDTH, filter_);
		if (multiplier != 1.0)
		{
			for (uint i = 0; i < FILTER_WIDTH*FILTER_WIDTH; ++i)
			{
				filter_[i] *= multiplier;
			}
		}
	}

	template<typename T, int FILTER_WIDTH>
	CPU void Filter<T, FILTER_WIDTH>::copyWholeFilterToDeviceMemory(void * destination) const
	{
		checkCudaErrors(cudaMemcpy(destination, this, sizeof(Filter<T, FILTER_WIDTH>), cudaMemcpyHostToDevice));
	}



}
