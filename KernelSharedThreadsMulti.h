#pragma once
#include "Runnable.h"
#include <vector>
#include <memory>
#include <string>
#include "MemoryPoolPinned.h"
#include "MemoryPoolPitched.h"

using namespace std;
namespace processing
{


	class KernelSharedThreadsMulti : public MultiRunnable
	{

	public:

		KernelSharedThreadsMulti();

		DELETECOPYASSINGMENT(KernelSharedThreadsMulti)

		virtual void run(vector<shared_ptr<ImageFactory>>& images, vector<vector<shared_ptr<Filter>>>& filters, vector<shared_ptr<float>>& results) override;

		virtual string getDescription() override
		{
			return MultiRunnable::getDescription() +  " GPU shared memmory multi threads";
		}

		



	private:

		
	};

	namespace KernelSharedThreadsMultiNS
	{
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
			bool makeZeros = false;
			bool goToPostprocess = false;
			int filterGroupStartIndex;


		};

		struct BatchGroupInfo {

			BatchGroupInfo(short filterWidth, short filterCount) :
				filterWidth_(filterWidth), 
				filterCount_(filterCount)
			{}

			short filterWidth_;
			short filterCount_;

		};

		struct BatchGroup
		{
			int filterCount_;
			int filterStart_;
			int filterEnd_;
			vector<BatchGroupInfo> filterInfos_;
		};

	}

}