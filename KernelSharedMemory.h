#pragma once
#include "Runnable.h"
#include <vector>
#include "Filter.h"
#include <memory>
#include <string>

using namespace std;
namespace processing
{

	template<typename T>
	class KernelSharedMemory : public Runnable<T>
	{
		static_assert(std::is_floating_point<T>::value, "Class KernelSlowConvolution can only be instantiazed with float, double or long double");
	public:

		KernelSharedMemory();

		DELETECOPYASSINGMENT(KernelSharedMemory<T>)

		virtual void run(ImageFactory& image, vector<shared_ptr<AbstractFilter<T>>>& filters, vector<shared_ptr<T>>& results)  override;

		virtual string getDescription() override 
		{
			return "GPU shared memmory";
		}

	private:

	};




}



