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

	class KernelSharedMemoryFullBlock : public Runnable
	{

	public:

		KernelSharedMemoryFullBlock();

		DELETECOPYASSINGMENT(KernelSharedMemoryFullBlock)

		virtual void run(ImageFactory& image, vector<shared_ptr<Filter>>& filters, vector<shared_ptr<float>>& results)  override;

		virtual string getDescription() override 
		{
			return "GPU shared memmory";
		}

	private:

	};




}



