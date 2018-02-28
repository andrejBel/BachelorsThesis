#pragma once
#include "Runnable.h"
#include <vector>
#include "Filter.h"
#include <memory>

using namespace std;

namespace processing 
{

	template<typename T>
	class CpuSlowConvolution : public Runnable<T>
	{
		static_assert(std::is_floating_point<T>::value, "Class KernelSlowConvolution can only be instantiazed with float, double or long double");
	public:

		CpuSlowConvolution();

		DELETECOPYASSINGMENT(CpuSlowConvolution<T>)

		virtual void run(ImageFactory& image, vector<shared_ptr<AbstractFilter<T>>>& filters, vector<shared_ptr<T>>& results) override;

		virtual string getDescription() override
		{
			return "CPU single core";
		}

	private:

		__host__ __forceinline__ int min(int a, int b);
		

		__host__ __forceinline__ int max(int a, int b);

		template <typename int FILTER_WIDTH>
		void convolution(ImageFactory& image, Filter<T, FILTER_WIDTH> * filter, T* outputImage);

	};



	template<typename T>
	template<typename int FILTER_WIDTH>
	inline void CpuSlowConvolution<T>::convolution(ImageFactory & image, Filter<T, FILTER_WIDTH>* filter, T * outputImage)
	{
		auto columns = image.getNumCols();
		auto rows = image.getNumRows();
		uchar * inputImage = image.getInputGrayPointer();
		int2 pointPosition;
		auto filterV = filter->getFilter();
		T result(0.0);
		for (int row = 0; row < rows; ++row)
		{
			for (int column = 0; column < columns; ++column)
			{
				result = 0.0;
				for (int y = 0; y < FILTER_WIDTH; ++y)
				{
					for (int x = 0; x < FILTER_WIDTH; ++x)
					{
						pointPosition.x = column + x - (FILTER_WIDTH / 2);
						pointPosition.y = row + y - (FILTER_WIDTH / 2);
						pointPosition.x = min(max(pointPosition.x, 0), columns - 1);
						pointPosition.y = min(max(pointPosition.y, 0), rows - 1);
						result += filterV[y * FILTER_WIDTH + x] * inputImage[pointPosition.y * columns + pointPosition.x];
					}
				}
				outputImage[row * columns + column] = result;
			}
		}
	}

}



