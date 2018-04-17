#pragma once
#include "Runnable.h"
#include <vector>
#include "Filter.h"
#include "processing.h"
#include <memory>

using namespace std;
namespace processing
{

	class KernelNaiveImproved : public SimpleRunnable
	{

	public:

		KernelNaiveImproved();

		DELETECOPYASSINGMENT(KernelNaiveImproved)

		virtual void run(vector<shared_ptr<ImageFactory>>& images, vector<shared_ptr<Filter>>& filters, vector<shared_ptr<float>>& results) override;

		virtual string getDescription() override
		{
			return SimpleRunnable::getDescription() + " GPU naive improved";
		}

	private:
		__host__ __forceinline__ int indexToCopyToMirrored(int index, int numCols, int numRows, const int filterWidth);


	};


	

}



