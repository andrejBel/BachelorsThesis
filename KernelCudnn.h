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


	class KernelCudnn : public SimpleRunnable
	{

	public:

		KernelCudnn();

		DELETECOPYASSINGMENT(KernelCudnn)

		virtual void run(vector<shared_ptr<ImageFactory>>& images, vector<shared_ptr<Filter>>& filters, vector<shared_ptr<float>>& results) override;

		virtual string getDescription() override
		{
			return SimpleRunnable::getDescription() +  " GPU Cudnn";
		}

	private:

	};


}



