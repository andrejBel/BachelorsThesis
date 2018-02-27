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

#define CPU __host__
#define CPUGPU __device__ __host__
#define CPUGPUINLINE __device__ __host__ __forceinline__ 

using namespace std;
namespace processing 
{
	template <typename T>
	class AbstractFilter
	{
	public:	
	
		CPUGPU AbstractFilter() {}

		CPUGPUINLINE virtual int getWidth() const = 0;

		CPUGPUINLINE virtual uint getSize() const = 0;

		CPU virtual void copyWholeFilterToDeviceMemory(void * destination) const = 0;

		CPU virtual ~AbstractFilter() {};
	};


	template <typename T, int FILTER_WIDTH>
	class Filter : public AbstractFilter<T>
	{
		static_assert(std::is_floating_point<T>::value, "Class Filter can only be instantiazed with float, double or long double");
	public:
		CPU Filter(vector<T> filter, const T multiplier = 1.0);

		CPU Filter(T* filter, const T multiplier = 1.0);

		CPUGPUINLINE const T* getFilter()
		{
			return filter_;
		}

		CPUGPUINLINE virtual int getWidth() const override
		{
			return FILTER_WIDTH;
		}

		CPU virtual void copyWholeFilterToDeviceMemory(void * destination) const override;

		CPUGPUINLINE virtual uint getSize() const override
		{
			return sizeof(Filter<T, FILTER_WIDTH>);
		}

		CPU virtual ~Filter() 
		{};

	private:

		T filter_[FILTER_WIDTH * FILTER_WIDTH];

	};

	
	template<typename T>
	shared_ptr<AbstractFilter<T>> createFilter(uint width, vector<T> filter, const T multiplier = 1.0);

	template<typename T>
	shared_ptr<AbstractFilter<T>> createFilter(uint width, T* filter, const T multiplier = 1.0);


	template<typename T>
	inline shared_ptr<AbstractFilter<T>> createFilter(uint width, vector<T> filter, const T  multiplier)
	{
		switch (width)
		{
		case 3: return make_shared<Filter<T, 3>>(filter, multiplier);
		case 5: return make_shared<Filter<T, 5>>(filter, multiplier);
		case 7: return make_shared<Filter<T, 7>>(filter, multiplier);
		default:
			break;
		}
		return shared_ptr<AbstractFilter<T>>();
	}

	template<typename T>
	inline shared_ptr<AbstractFilter<T>> createFilter(uint width, T * filter, const T  multiplier)
	{
		switch (width)
		{
		case 3: return make_shared<Filter<T, 3>>(filter, multiplier);
		case 5: return make_shared<Filter<T, 5>>(filter, multiplier);
		case 7: return make_shared<Filter<T, 7>>(filter, multiplier);
		default:
			break;
		}
		return shared_ptr<AbstractFilter<T>>();
	}


}




#endif


