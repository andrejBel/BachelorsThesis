#include "CpuCroppedMulti.h"



#include <vector>
#include <memory>
#include <algorithm>
#include <iostream>



#define ACCEPTFILTER(FILTERWIDTH)\
case FILTERWIDTH:\
{\
	convolution<FILTERWIDTH>(*image.get(), filter->getFilter(), partialResults[i].get()); \
	break;\
}

namespace processing
{

	CpuCroppedMulti::CpuCroppedMulti() : MultiRunnable()
	{
	}

	void CpuCroppedMulti::run(vector<shared_ptr<ImageFactory>>& images, vector<vector<shared_ptr<Filter>>>& filters, vector<shared_ptr<float>>& results)
	{
		size_t imageSize = images.size();
		int numCols = images[0]->getNumCols(); //x
		int numRows = images[0]->getNumRows(); //y
		
		size_t pixels = numCols * numRows;

		vector<shared_ptr<float>> partialResults(imageSize);
		for (size_t i = 0; i < imageSize; ++i)
		{
			partialResults[i] = makeArray<float>(pixels);
		}
		for (auto& filterGroup : filters)
		{
			for (size_t i = 0; i < imageSize; ++i)
			{
				std::fill(partialResults[i].get(), partialResults[i].get() + pixels, 0);
			}
			const int filterWidth = filterGroup[0]->getWidth();
			for (size_t i = 0; i < imageSize; i++)
			{
				shared_ptr<ImageFactory> image = images[i];
				shared_ptr<Filter> filter = filterGroup[i];
				switch (filterWidth)
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
			}
			shared_ptr<float> result = makeArray<float>(pixels);
			float* resultPtr = result.get();
			std::fill(result.get(), result.get() + pixels, 0);
			for (size_t j = 0; j < pixels; j++)
			{
				for (size_t k = 0; k < imageSize; k++)
				{
					resultPtr[j] += partialResults[k].get()[j];
				}
			}

			results.push_back(result);
		}


	}


	__host__ __forceinline__ int CpuCroppedMulti::min(int a, int b)
	{
		return a < b ? a : b;
	}

	__host__ __forceinline__ int CpuCroppedMulti::max(int a, int b)
	{
		return a > b ? a : b;
	}



}


