#pragma once
#include "Runnable.h"
#include <vector>
#include "Filter.h"
#include <memory>
#include "ThreadPool.h"

using namespace std;
namespace processing
{

	template<typename T>
	class KernelSlowConvolutionNoEdgeCopy : public Runnable<T>
	{
		static_assert(std::is_floating_point<T>::value, "Class KernelSlowConvolution can only be instantiazed with float, double or long double");
	public:

		KernelSlowConvolutionNoEdgeCopy(vector< shared_ptr<AbstractFilter <T>> >& filters);

		DELETECOPYASSINGMENT(KernelSlowConvolutionNoEdgeCopy<T>)

			virtual void run(ImageFactory& image, vector<shared_ptr<T>>& results)  override;

	private:
		vector< shared_ptr<AbstractFilter <T>> >& h_filters_;
		ThreadPool threadPool_;
		__host__ __forceinline__ int indexToCopyToMirrored(int index, int numCols, int numRows, const int filterWidth);


	};

	__device__ __forceinline__ size_t indexInNew(int indexX, int indexY, int originalWidth, int originalHeight, int filterWidth);


	

}



