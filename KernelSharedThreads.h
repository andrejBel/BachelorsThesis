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

	class KernelSharedThreads : public SimpleRunnable
	{

	public:

		KernelSharedThreads();

		DELETECOPYASSINGMENT(KernelSharedThreads)

		virtual void run(vector<shared_ptr<ImageFactory>>& images, vector<shared_ptr<Filter>>& filters, vector<shared_ptr<float>>& results) override;

		virtual string getDescription() override
		{
			return SimpleRunnable::getDescription() +  " GPU shared memmory threads";
		}

	private:

	};

	struct Job {

		shared_ptr<float> filters_;
		int filterCount_;
		int bufferStart_;
		int filterWidth_;
		bool returnInputImage_ = false;
		bool finish_ = false;
		float* inputImage_;
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



