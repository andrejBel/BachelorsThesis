#include "CpuSlowConvolution.h"
#include "processing.h"
#include <vector>
#include <memory>
#include "Filter.h"
#include <algorithm>
#include <iostream>

#define ACCEPTFILTER(FILTERWIDTH)\
case FILTERWIDTH:\
{\
	Filter<T, FILTERWIDTH> * ptr = (Filter<T, FILTERWIDTH> *) (filter.get());\
	convolution(image, ptr, result.get());\
	break;\
}

namespace processing 
{
	template<typename T>
	CpuSlowConvolution<T>::CpuSlowConvolution()
	{
	}

	template<typename T>
	void CpuSlowConvolution<T>::run(ImageFactory& image, vector<shared_ptr<AbstractFilter<T>>>& filters, vector<shared_ptr<T>>& results)
	{
		for (auto& filter : filters)
		{
			shared_ptr<T> result = makeArray<T>(image.getNumPixels());
			switch (filter->getWidth())
			{
			ACCEPTFILTER(1)
			ACCEPTFILTER(3)
			ACCEPTFILTER(5)
			ACCEPTFILTER(7)
			ACCEPTFILTER(9)
			ACCEPTFILTER(11)
			ACCEPTFILTER(13)
			ACCEPTFILTER(15)
			default:
				std::cerr << "Filter with width: " << filter->getWidth() << " not supported!" << endl;
				break;
			}
			results.push_back(result);
		}
	}

	template<typename T>
	__host__ __forceinline__ int CpuSlowConvolution<T>::min(int a, int b)
	{
		return a < b ? a : b;
	}

	template<typename T>
	__host__ __forceinline__ int CpuSlowConvolution<T>::max(int a, int b)
	{
		return a > b ? a : b;
	}
	


}


