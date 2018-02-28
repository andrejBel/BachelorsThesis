#pragma once
#include "Runnable.h"
#include <vector>
#include "Filter.h"
#include <memory>

using namespace std;
namespace processing
{

	template<typename T>
	class KernelSlowConvolutionNoEdgeCopy : public Runnable<T>
	{
		static_assert(std::is_floating_point<T>::value, "Class KernelSlowConvolution can only be instantiazed with float, double or long double");
	public:

		KernelSlowConvolutionNoEdgeCopy();

		DELETECOPYASSINGMENT(KernelSlowConvolutionNoEdgeCopy<T>)

		virtual void run(ImageFactory& image, vector<shared_ptr<AbstractFilter<T>>>& filters, vector<shared_ptr<T>>& results) override;

		virtual string getDescription() override
		{
			return "CPU slow without edge copiing";
		}

	private:
		__host__ __forceinline__ int indexToCopyToMirrored(int index, int numCols, int numRows, const int filterWidth);


	};


	

}



