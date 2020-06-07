#include "CpuCropped.h"


#include <vector>
#include <memory>
#include <algorithm>
#include <iostream>


#define ACCEPTFILTER(FILTERWIDTH)\
case FILTERWIDTH:\
{\
	result = makeArray<float>((image->getNumCols() - (FILTERWIDTH - 1)) * (image->getNumRows() - (FILTERWIDTH - 1)));\
	convolution<FILTERWIDTH>(*image.get(), filter->getFilter(), result.get());\
	break;\
}

namespace processing
{

	CpuCropped::CpuCropped() :
		SimpleRunnable(true)
	{
	}

	void CpuCropped::run(vector<shared_ptr<ImageFactory>>& images, vector<shared_ptr<Filter>>& filters, vector<shared_ptr<float>>& results)
	{
		for (auto& image : images)
		{
			for (auto& filter : filters)
			{
				shared_ptr<float> result;
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
	}

	__host__ __forceinline__ int CpuCropped::min(int a, int b)
	{
		return a < b ? a : b;
	}

	__host__ __forceinline__ int CpuCropped::max(int a, int b)
	{
		return a > b ? a : b;
	}



}


