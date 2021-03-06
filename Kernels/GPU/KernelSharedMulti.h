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


	class KernelSharedMulti : public MultiRunnable
	{

	public:

		KernelSharedMulti();

		DELETECOPYASSINGMENT(KernelSharedMulti)

		virtual void run(vector<shared_ptr<ImageFactory>>& images, vector<vector<shared_ptr<Filter>>>& filters, vector<shared_ptr<float>>& results) override;

		virtual string getDescription() override
		{
			return MultiRunnable::getDescription() + " GPU shared memmory multi";
		}

	private:

	};


}