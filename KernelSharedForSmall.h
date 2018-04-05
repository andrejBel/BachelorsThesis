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

	
	class KernelSharedForSmall : public Runnable
	{
		
	public:

		KernelSharedForSmall();

		DELETECOPYASSINGMENT(KernelSharedForSmall)

		virtual void run(ImageFactory& image, vector<shared_ptr<Filter>>& filters, vector<shared_ptr<float>>& results)  override;

		virtual string getDescription() override
		{
			return "GPU shared memmory for small";
		}

	private:

	};


}



