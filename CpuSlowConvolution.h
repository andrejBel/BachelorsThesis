#pragma once
#include "Runnable.h"
#include <vector>
#include "Filter.h"
#include "processing.h"
#include <memory>

using namespace std;

namespace processing 
{

	class CpuSlowConvolution : public Runnable
	{

	public:

		CpuSlowConvolution();

		DELETECOPYASSINGMENT(CpuSlowConvolution)

		virtual void run(ImageFactory& image, vector<shared_ptr<Filter>>& filters, vector<shared_ptr<float>>& results) override;

		virtual string getDescription() override
		{
			return "CPU single core";
		}

	private:

		__host__ __forceinline__ int min(int a, int b);
		

		__host__ __forceinline__ int max(int a, int b);

		template <typename int FILTER_WIDTH>
		void convolution(ImageFactory& image, const float * filter, float* outputImage);

	};


	template<typename int FILTER_WIDTH>
	inline void CpuSlowConvolution::convolution(ImageFactory & image, const float* filter, float * outputImage)
	{
		auto columns = image.getNumCols();
		auto rows = image.getNumRows();
		const float* inputImage = image.getInputGrayPointerFloat();
		int2 pointPosition;
		float result(0.0);
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
						result += filter[y * FILTER_WIDTH + x] * inputImage[pointPosition.y * columns + pointPosition.x];
					}
				}
				outputImage[row * columns + column] = result;
			}
		}
	}

}



