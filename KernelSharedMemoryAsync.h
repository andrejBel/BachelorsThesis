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


	class KernelSharedMemoryAsync : public Runnable
	{

	public:

		KernelSharedMemoryAsync();

		DELETECOPYASSINGMENT(KernelSharedMemoryAsync)

		virtual void run(ImageFactory& image, vector<shared_ptr<AbstractFilter>>& filters, vector<shared_ptr<float>>& results)  override;

		virtual string getDescription() override
		{
			return "GPU shared memmory async";
		}

	private:

		ThreadPool threadpool_;


	};

}



