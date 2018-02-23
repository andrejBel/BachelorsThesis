#include "Filter.h"

#include <cuda_runtime.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <processing.h>

namespace processing 
{

	template<typename T>
	Filter<T>::Filter(uint width, uint height, vector<T> filter, const T multiplier) :
		multiplier_(multiplier),
		width_(width),
		height_(height),
		d_filter_(nullptr),
		h_filter_(filter)
	{
		h_filter_.push_back(multiplier_);
	}

	template<typename T>
	Filter<T>::Filter(uint width, uint height, const T * filter, const T multiplier):
		multiplier_(multiplier),
		width_(width),
		height_(height),
		d_filter_(nullptr),
		h_filter_(width * height + 1)
	{
		for (size_t index = 0; index < width * height; index++)
		{
			h_filter_[index] = filter[index];
		}
		h_filter_[width * height] = multiplier;
	}

	template<typename T>
	T * Filter<T>::getHostFilterPointer()
	{
		return h_filter_.data();
	}

	template<typename T>
	T * Filter<T>::getDeviceFilterPointer()
	{
		return d_filter_;
	}

	template<typename T>
	void Filter<T>::allocateAndCopyHostFilterToDevice()
	{
		d_filter_ = allocateMemmoryDevice<T>(h_filter_.size() * sizeof(T));
		checkCudaErrors(cudaMemcpy(d_filter_, h_filter_.data(), h_filter_.size() * sizeof(T), cudaMemcpyHostToDevice));
	}

	template<typename T>
	void Filter<T>::deallocateDeviceFilter()
	{
		deallocateMemmoryDevice(d_filter_);
		d_filter_ = nullptr;
	}

	template<typename T>
	T Filter<T>::getMultiplier()
	{
		return multiplier_;
	}

	template<typename T>
	Filter<T>::~Filter()
	{
		deallocateMemmoryDevice(d_filter_);
	}


}
