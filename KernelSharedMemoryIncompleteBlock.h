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


	class KernelSharedMemoryIncompleteBlock : public Runnable
	{

	public:

		KernelSharedMemoryIncompleteBlock();

		DELETECOPYASSINGMENT(KernelSharedMemoryIncompleteBlock)

		virtual void run(ImageFactory& image, vector<shared_ptr<Filter>>& filters, vector<shared_ptr<float>>& results)  override;

		virtual string getDescription() override
		{
			return "GPU shared memmory incomplete block";
		}

	private:

		ThreadPool threadpool_;


	};

}



