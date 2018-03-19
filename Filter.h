#ifndef FILTER_H
#define FILTER_H

#include <opencv2/core/core.hpp>
#include <vector>
#include <type_traits>

#include <cuda_runtime.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <iostream>

#define CPU __host__
#define CPUGPU __device__ __host__
#define CPUGPUINLINE __device__ __host__ __forceinline__ 

using namespace std;
namespace processing 
{
	class AbstractFilter
	{
	public:	
	
		CPUGPU AbstractFilter() {}

		CPUGPUINLINE virtual int getWidth() const = 0;

		CPUGPUINLINE virtual int getSize() const = 0;

		CPUGPUINLINE virtual const float* getFilter() const = 0;

		CPU virtual ~AbstractFilter() {};
	};


	template <int FILTER_WIDTH>
	class Filter : public AbstractFilter
	{

	public:
		
		
		Filter(vector<float> filter, const float multiplier = 1.0f)
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

		Filter(float * filter, const float multiplier = 1.0f)
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

		CPUGPUINLINE const float* getFilter() const override
		{
			return filter_;
		}

		CPUGPUINLINE virtual int getWidth() const override
		{
			return FILTER_WIDTH;
		}


		CPUGPUINLINE virtual int getSize() const override
		{
			return FILTER_WIDTH * FILTER_WIDTH;
		}

		CPU virtual ~Filter() 
		{};
		
	private:

		float filter_[FILTER_WIDTH * FILTER_WIDTH];

	};


	


}




#endif


