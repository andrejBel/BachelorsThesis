#pragma once
#include "Runnable.h"
#include <vector>
#include "Filter.h"
#include <memory>
#include <string>
#include "ThreadPool.h"

using namespace std;
namespace processing
{
	static const int THREADS_NUMBER = 4;

	template<typename T>
	class KernelSharedMemoryManaged : public Runnable<T>
	{
		static_assert(std::is_floating_point<T>::value, "Class KernelSlowConvolution can only be instantiazed with float, double or long double");
	public:

		KernelSharedMemoryManaged();

		DELETECOPYASSINGMENT(KernelSharedMemoryManaged<T>)

		virtual void run(ImageFactory& image, vector<shared_ptr<AbstractFilter<T>>>& filters, vector<shared_ptr<T>>& results)  override;

		virtual string getDescription() override
		{
			return "GPU shared memmory managed";
		}

	private:

	};




}



