#include "CpuSlowConvolution.h"

#include <vector>
#include <memory>
#include <algorithm>
#include <iostream>


#define ACCEPTFILTER(FILTERWIDTH)\
case FILTERWIDTH:\
{\
	convolution<FILTERWIDTH>(image, filter->getFilter(), result.get());\
	break;\
}

namespace processing 
{

	CpuSlowConvolution::CpuSlowConvolution()
	{
	}

	void CpuSlowConvolution::run(ImageFactory& image, vector<shared_ptr<Filter>>& filters, vector<shared_ptr<float>>& results)
	{
		for (auto& filter : filters)
		{
			shared_ptr<float> result = makeArray<float>(image.getNumPixels());
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

	__host__ __forceinline__ int CpuSlowConvolution::min(int a, int b)
	{
		return a < b ? a : b;
	}

	__host__ __forceinline__ int CpuSlowConvolution::max(int a, int b)
	{
		return a > b ? a : b;
	}
	


}


