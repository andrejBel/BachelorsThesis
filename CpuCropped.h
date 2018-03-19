#pragma once
#include "Runnable.h"
#include <vector>
#include "Filter.h"
#include "processing.h"
#include <memory>

using namespace std;

namespace processing
{

	
	class CpuCropped : public Runnable
	{

	public:

		CpuCropped();

		DELETECOPYASSINGMENT(CpuCropped)

		virtual void run(ImageFactory& image, vector<shared_ptr<AbstractFilter>>& filters, vector<shared_ptr<float>>& results) override;

		virtual string getDescription() override
		{
			return "CPU cropped single core";
		}

	private:

		__host__ __forceinline__ int min(int a, int b);


		__host__ __forceinline__ int max(int a, int b);

		template <typename int FILTER_WIDTH>
		void convolution(ImageFactory& image, Filter<FILTER_WIDTH> * filter, float* outputImage);

	};



	
	template<typename int FILTER_WIDTH>
	inline void CpuCropped::convolution(ImageFactory & image, Filter<FILTER_WIDTH>* filter, float * outputImage)
	{
		auto columns = image.getNumCols();
		auto rows = image.getNumRows();
		const float * inputImage = image.getInputGrayPointerFloat();
		int2 pointPosition;
		auto filterV = filter->getFilter();
		float result(0.0);
		for (int row = (FILTER_WIDTH / 2); row < rows - (FILTER_WIDTH / 2); ++row)
		{
			for (int column = (FILTER_WIDTH / 2); column < columns - (FILTER_WIDTH / 2); ++column)
			{
				result = 0.0;
				for (int y = 0; y < FILTER_WIDTH; ++y)
				{
					for (int x = 0; x < FILTER_WIDTH; ++x)
					{
						pointPosition.x = column + x - (FILTER_WIDTH / 2);
						pointPosition.y = row + y - (FILTER_WIDTH / 2);
						result += filterV[y * FILTER_WIDTH + x] * inputImage[pointPosition.y * columns + pointPosition.x];
					}
				}
				outputImage[(row - (FILTER_WIDTH / 2) ) * (columns - (FILTER_WIDTH - 1)) + (column - (FILTER_WIDTH / 2))] = result;
			}
		}
	}

}



