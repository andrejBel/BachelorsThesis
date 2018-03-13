#include "CpuCropped.h"


#include <vector>
#include <memory>
#include "Filter.h"
#include <algorithm>
#include <iostream>

#define ACCEPTFILTER(FILTERWIDTH)\
case FILTERWIDTH:\
{\
	Filter<T, FILTERWIDTH> * ptr = (Filter<T, FILTERWIDTH> *) (filter.get());\
	result = makeArray<T>((image.getNumCols() - (FILTERWIDTH - 1)) * (image.getNumRows() - (FILTERWIDTH - 1)));\
	convolution(image, ptr, result.get());\
	break;\
}

namespace processing
{
	template<typename T>
	CpuCropped<T>::CpuCropped()
	{
	}

	template<typename T>
	void CpuCropped<T>::run(ImageFactory& image, vector<shared_ptr<AbstractFilter<T>>>& filters, vector<shared_ptr<T>>& results)
	{
		for (auto& filter : filters)
		{
			shared_ptr<T> result;
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
	__host__ __forceinline__ int CpuCropped<T>::min(int a, int b)
	{
		return a < b ? a : b;
	}

	template<typename T>
	__host__ __forceinline__ int CpuCropped<T>::max(int a, int b)
	{
		return a > b ? a : b;
	}



}


