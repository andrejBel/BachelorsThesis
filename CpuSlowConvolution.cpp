#include "CpuSlowConvolution.h"
#include "processing.h"
#include <vector>
#include <memory>
#include "Filter.h"


template <typename T, typename int N>
void func(processing::ImageFactory& image, processing::Filter<T, N> * filter) 
{
	auto columns = image.getNumCols();
	auto rows = image.getNumRows();
	for (size_t index1D = 0; index1D < columns * rows; index1D++)
	{
		for (int y = 0; y < N; ++y)
		{
			for (int x = 0; x < N; ++x)
			{
				filter->filter_[y * N + x];
			}
		}
	}
}

namespace processing 
{

	template<typename T>
	CpuSlowConvolution<T>::CpuSlowConvolution(vector<shared_ptr<AbstractFilter<T>>>& filters):
		filters_(filters)
	{
	}

	template<typename T>
	void CpuSlowConvolution<T>::run(ImageFactory & image)
	{
		for (auto& filter : filters_)
		{
			switch (filter->getWidth())
			{
			case 3:
			{
				Filter<T, 3> * ptr = (Filter<T, 3> *) (filter.get());
				func<T, 3>(image, ptr);
				break;
			}
			case 5:
			{
				Filter<T, 5> * ptr = (Filter<T, 5> *) (filter.get());
				func<T, 5>(image, ptr);
				break;
			}
			case 7:
			{
				Filter<T, 7> * ptr = (Filter<T, 7> *) (filter.get());
				func<T,7>(image, ptr);
				break;
			}
			default:
				break;
			}
		}
	}

}


