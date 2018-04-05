#pragma once
#include "Runnable.h"
#include <vector>
#include "Filter.h"
#include "processing.h"
#include <memory>

using namespace std;
namespace processing
{

	class KernelSlowConvolutionNoEdgeCopy : public Runnable
	{

	public:

		KernelSlowConvolutionNoEdgeCopy();

		DELETECOPYASSINGMENT(KernelSlowConvolutionNoEdgeCopy)

		virtual void run(ImageFactory& image, vector<shared_ptr<Filter>>& filters, vector<shared_ptr<float>>& results) override;

		virtual string getDescription() override
		{
			return "CPU slow without edge copiing";
		}

	private:
		__host__ __forceinline__ int indexToCopyToMirrored(int index, int numCols, int numRows, const int filterWidth);


	};


	

}



