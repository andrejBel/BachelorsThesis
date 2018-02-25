#include "CpuSlowConvolution.h"
#include "processing.h"
#include <vector>
#include <memory>
#include "Filter.h"
#include <algorithm>



namespace processing 
{


	template <typename T, typename int WIDTH>
	void convolution(ImageFactory& image, Filter<T, WIDTH> * filter, T* outputImage)
	{
		auto columns = image.getNumCols();
		auto rows = image.getNumRows();
		uchar * inputImage = image.getInputGrayPointer();
		int2 absoluteImagePosition;
		int2 pointPosition;
		auto filterV = filter->getFilter();
		T result(0.0);
		for (int row = 0; row < rows; ++row)
		{
			for (int column = 0; column < columns; ++column)
			{
				result = 0.0;
				for (int y = 0; y < WIDTH; ++y)
				{
					for (int x = 0; x < WIDTH; ++x)
					{
						pointPosition.x = column + x - (WIDTH / 2);
						pointPosition.y = row + y - (WIDTH / 2);
						pointPosition.x = std::min(std::max(pointPosition.x, 0), columns - 1);
						pointPosition.y = std::min(std::max(pointPosition.y, 0), rows - 1);
						result += filterV[y * WIDTH + x] * inputImage[pointPosition.y * columns + pointPosition.x];
					}
				}
				outputImage[row * columns + column] = result;
			}
		}
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


}


