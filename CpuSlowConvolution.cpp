#include "CpuSlowConvolution.h"
#include "processing.h"
#include <vector>
#include <memory>
#include "Filter.h"
#include <algorithm>



namespace processing 
{


	template <typename T, typename int FILTER_WIDTH>
	void convolution(ImageFactory& image, Filter<T, FILTER_WIDTH> * filter, T* outputImage)
	{
		
	}

	template<typename T>
	CpuSlowConvolution<T>::CpuSlowConvolution(vector<shared_ptr<AbstractFilter<T>>>& filters):
		filters_(filters)
	{
	}

	template<typename T>
	void CpuSlowConvolution<T>::run(ImageFactory & image, vector<shared_ptr<T>>& results)
	{
		for (auto& filter : filters_)
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
			results.push_back(result);
			//image.saveGrayImgOut("blurredImage.jpg");
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


