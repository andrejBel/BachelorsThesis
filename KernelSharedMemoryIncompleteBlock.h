#pragma once
#include "Runnable.h"
#include <vector>
#include "Filter.h"
#include "processing.h"
#include <memory>
#include <string>
#include "ThreadPool.h"
using namespace std;
namespace processing
{


	class KernelSharedMemoryIncompleteBlock : public SimpleRunnable
	{

	public:

		KernelSharedMemoryIncompleteBlock();

		DELETECOPYASSINGMENT(KernelSharedMemoryIncompleteBlock)

		virtual void run(vector<shared_ptr<ImageFactory>>& images, vector<shared_ptr<Filter>>& filters, vector<shared_ptr<float>>& results) override;

		virtual string getDescription() override
		{
			return SimpleRunnable::getDescription() +  " GPU shared memmory incomplete block";
		}

	private:

		ThreadPool threadpool_;


	};

}



