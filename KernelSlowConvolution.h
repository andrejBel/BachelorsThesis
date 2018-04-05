#pragma once
#include "Runnable.h"
#include <vector>
#include "Filter.h"
#include "processing.h"
#include <memory>

using namespace std;
namespace processing 
{


	class KernelSlowConvolution : public Runnable
	{

	public:

		KernelSlowConvolution();

		DELETECOPYASSINGMENT(KernelSlowConvolution)

		virtual void run(ImageFactory& image, vector<shared_ptr<Filter>>& filters, vector<shared_ptr<float>>& results)  override;

		virtual string getDescription() override
		{
			return "CPU slow with edge copiing";
		}

	private:

	    __host__ __forceinline__ int indexToCopyToMirrored(int index, int numCols, int numRows, const int filterWidth);


	};

	__device__ __forceinline__ int indexInNew(int indexX, int indexY, int originalWidth, int originalHeight, int filterWidth);


}



