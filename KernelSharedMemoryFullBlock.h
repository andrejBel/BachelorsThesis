#pragma once
#include "Runnable.h"
#include <vector>
#include "Filter.h"
#include "processing.h"
#include <memory>
#include <string>

using namespace std;
namespace processing
{

	class KernelSharedMemoryFullBlock : public SimpleRunnable
	{

	public:

		KernelSharedMemoryFullBlock();

		DELETECOPYASSINGMENT(KernelSharedMemoryFullBlock)

		virtual void run(vector<shared_ptr<ImageFactory>>& images, vector<shared_ptr<Filter>>& filters, vector<shared_ptr<float>>& results) override;

		virtual string getDescription() override 
		{
			return SimpleRunnable::getDescription() +  " GPU shared memory full block";
		}

	private:

	};




}



