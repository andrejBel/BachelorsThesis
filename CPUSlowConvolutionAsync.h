#pragma once
#include "Runnable.h"
#include <vector>
#include "processing.h"
#include "Filter.h"
#include <memory>
#include "ThreadPool.h"

using namespace std;

namespace processing
{

	
	class CPUSlowConvolutionAsync : public Runnable
	{
		
	public:

		CPUSlowConvolutionAsync();

		DELETECOPYASSINGMENT(CPUSlowConvolutionAsync)

		virtual void run(ImageFactory& image, vector<shared_ptr<AbstractFilter>>& filters, vector<shared_ptr<float>>& results) override;

		virtual string getDescription() override
		{
			return "CPU multiple core";
		}

		template <typename int FILTER_WIDTH>
		void divideAndConquer(size_t leftBorder, size_t rightBorder, int numCols, int numRows, Filter<FILTER_WIDTH>* filter, const float* inputImage, float* outputImage);

	private:
		ThreadPool threadPool_;

		__host__ __forceinline__ int min(int a, int b);

		__host__ __forceinline__ int max(int a, int b);

		template <typename int FILTER_WIDTH>
		void convolution(ImageFactory& image, Filter<FILTER_WIDTH> * filter, float* outputImage);

		static const unsigned int NUMBER_OF_THREADS = 4;

	};


	template<typename int FILTER_WIDTH>
	inline void CPUSlowConvolutionAsync::convolution(ImageFactory & image, Filter<FILTER_WIDTH>* filter, float * outputImage)
	{
		auto columns = image.getNumCols();
		auto rows = image.getNumRows();
		auto pixels = image.getNumPixels();
		const float * inputImage = image.getInputGrayPointerFloat();
		
		double difference = pixels;
		difference = std::ceil(difference/NUMBER_OF_THREADS);
		size_t leftBorder = 0;
		size_t rightBorder = size_t(difference);
		for (uint i = 0; i < NUMBER_OF_THREADS; ++i)
		{
			threadPool_.addTask([this,leftBorder, rightBorder,columns, rows, filter, inputImage, outputImage]() 
			{
				this->divideAndConquer(leftBorder, rightBorder, columns, rows, filter, inputImage, outputImage);
			});
			leftBorder = rightBorder;
			rightBorder = std::min(rightBorder + size_t(difference), size_t(pixels));
		}

	}

	template<typename int FILTER_WIDTH>
	inline void CPUSlowConvolutionAsync::divideAndConquer(size_t leftBorder, size_t rightBorder, int numCols, int numRows, Filter<FILTER_WIDTH>* filter, const float* inputImage, float* outputImage)
	{
		int2 pointPosition;
		auto filterV = filter->getFilter();
		float result(0.0);
		int column; //x
		int row; //y
		for (size_t index1D = leftBorder; index1D < rightBorder; index1D++)
		{
			column = index1D % numCols;
			row = index1D / numCols;
			result = 0.0;
			for (int y = 0; y < FILTER_WIDTH; ++y)
			{
				for (int x = 0; x < FILTER_WIDTH; ++x)
				{
					pointPosition.x = column + x - (FILTER_WIDTH / 2);
					pointPosition.y = row + y - (FILTER_WIDTH / 2);
					pointPosition.x = min(max(pointPosition.x, 0), numCols - 1);
					pointPosition.y = min(max(pointPosition.y, 0), numRows - 1);
					result += filterV[y * FILTER_WIDTH + x] * inputImage[pointPosition.y * numCols + pointPosition.x];
				}
			}
			outputImage[index1D] = result;
		}
	}


}



