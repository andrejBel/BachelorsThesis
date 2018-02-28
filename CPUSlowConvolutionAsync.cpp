#include "CPUSlowConvolutionAsync.h"
#include "processing.h"
#include <vector>
#include <memory>
#include "Filter.h"
#include <algorithm>



namespace processing
{
	template<typename T>
	CPUSlowConvolutionAsync<T>::CPUSlowConvolutionAsync() :
		threadPool_(NUMBER_OF_THREADS)
	{
	}

	template<typename T>
	void CPUSlowConvolutionAsync<T>::run(ImageFactory& image, vector<shared_ptr<AbstractFilter<T>>>& filters, vector<shared_ptr<T>>& results)
	{
		for (auto& filter : filters)
		{
			shared_ptr<T> result = makeArray<T>(image.getNumPixels());
			switch (filter->getWidth())
			{
			case 3:
			{
				Filter<T, 3> * ptr = (Filter<T, 3> *) (filter.get());
				convolution(image, ptr, result.get());
				break;
			}
			case 5:
			{
				Filter<T, 5> * ptr = (Filter<T, 5> *) (filter.get());
				convolution(image, ptr, result.get());
				break;
			}
			case 7:
			{
				Filter<T, 7> * ptr = (Filter<T, 7> *) (filter.get());
				convolution(image, ptr, result.get());
				break;
			}
			default:
				break;
			}
			threadPool_.finishAll();
			results.push_back(result);
		}
	}

	template<typename T>
	__host__ __forceinline__ int CPUSlowConvolutionAsync<T>::min(int a, int b)
	{
		return a < b ? a : b;
	}

	template<typename T>
	__host__ __forceinline__ int CPUSlowConvolutionAsync<T>::max(int a, int b)
	{
		return a > b ? a : b;
	}



}



