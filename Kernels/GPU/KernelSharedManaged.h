#pragma once
#include "Runnable.h"
#include <vector>
#include "processing.h"
#include "Filter.h"
#include <memory>
#include <string>

using namespace std;
namespace processing
{

	class KernelSharedManaged : public SimpleRunnable
	{

	public:

		KernelSharedManaged();

		DELETECOPYASSINGMENT(KernelSharedManaged)

			virtual void run(vector<shared_ptr<ImageFactory>>& images, vector<shared_ptr<Filter>>& filters, vector<shared_ptr<float>>& results) override;

		virtual string getDescription() override
		{
			return SimpleRunnable::getDescription() + " GPU shared memmory managed";
		}

	private:

	};

	namespace NSKernelSharedManaged 
	{
	
		struct Job {

			shared_ptr<float> filters_;
			int filterCount_;
			int filterWidth_;
			bool finish_ = false;
			float* inputImage_;
			bool returnInputImage_ = false;
			int numRows;
			int numCols;

		};

		struct FilterBox
		{
			shared_ptr<float> memory_;
			int filterCount_;
			int filterWidth_;

		};

	}

	

}



