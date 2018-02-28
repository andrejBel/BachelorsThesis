#pragma once
#include "Runnable.h"
#include <vector>
#include "Filter.h"
#include <memory>

using namespace std;
namespace processing 
{

	template<typename T>
	class KernelSlowConvolution : public Runnable<T>
	{
		static_assert(std::is_floating_point<T>::value, "Class KernelSlowConvolution can only be instantiazed with float, double or long double");
	public:

		KernelSlowConvolution();

		DELETECOPYASSINGMENT(KernelSlowConvolution<T>)

		virtual void run(ImageFactory& image, vector<shared_ptr<AbstractFilter<T>>>& filters, vector<shared_ptr<T>>& results)  override;

		virtual string getDescription() override
		{
			return "CPU slow with edge copiing";
		}

	private:

	    __host__ __forceinline__ int indexToCopyToMirrored(int index, int numCols, int numRows, const int filterWidth);


	};

	__device__ __forceinline__ int indexInNew(int indexX, int indexY, int originalWidth, int originalHeight, int filterWidth);


}



