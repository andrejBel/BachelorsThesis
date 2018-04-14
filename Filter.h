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

	class Filter
	{

	public:
	
		Filter(const int filterWidth,const vector<float>& filter, const float multiplier = 1.0f) : 
			Filter(filterWidth)
		{
			std::copy(filter.begin(), filter.begin() + filterWidth_*filterWidth_, filter_.begin());
			if (multiplier != 1.0)
			{
				for (int i = 0; i < filterWidth_* filterWidth_; ++i)
				{
					filter_[i] *= multiplier;
				}
			}
		}

		Filter(const int filterWidth, float * filter, const float multiplier = 1.0f) :
			Filter(filterWidth)
		{
			std::copy(filter, filter + filterWidth_ * filterWidth_, filter_.begin());
			if (multiplier != 1.0)
			{
				for (int i = 0; i < filterWidth_*filterWidth_; ++i)
				{
					filter_[i] *= multiplier;
				}
			}
		}

		inline const float* getFilter() const
		{
			return filter_.data();
		}

		inline virtual int getWidth() const
		{
			return filterWidth_;
		}


		inline virtual int getSize() const
		{
			return filterWidth_ * filterWidth_;
		}

		~Filter() 
		{}
		
	private:
		Filter(const int filterWidth) :
			filterWidth_(filterWidth),
			filter_(filterWidth_ * filterWidth_)
		{}


		const int filterWidth_;
		vector<float> filter_;
		

	};


	


}




#endif


