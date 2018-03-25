#pragma once
#include "Runnable.h"
#include <vector>
#include "processing.h"
#include "Filter.h"
#include <memory>
#include <string>
#include "MemoryPoolPinned.h"
#include "MemoryPoolPitched.h"

using namespace std;
namespace processing
{


	class KernelSharedMulti : public Runnable
	{

	public:

		KernelSharedMulti();

		DELETECOPYASSINGMENT(KernelSharedMulti)

		virtual void run(ImageFactory& image, vector<shared_ptr<AbstractFilter>>& filters, vector<shared_ptr<float>>& results)  override;

		virtual void run(vector<shared_ptr<ImageFactory>>& images, vector<vector<shared_ptr<AbstractFilter>>>& filters, vector<shared_ptr<float>>& results) override;

		virtual string getDescription() override
		{
			return "GPU shared memmory multi";
		}

	private:

	};


}