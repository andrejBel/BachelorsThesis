#pragma once
#include "Runnable.h"
#include <vector>
#include "Filter.h"
#include <memory>
#include "ThreadPool.h"

using namespace std;
namespace processing
{

	template<typename T>
	class KernelSharedMemory : public Runnable<T>
	{
		static_assert(std::is_floating_point<T>::value, "Class KernelSlowConvolution can only be instantiazed with float, double or long double");
	public:

		KernelSharedMemory(vector< shared_ptr<AbstractFilter <T>> >& filters);

		DELETECOPYASSINGMENT(KernelSharedMemory<T>)

			virtual void run(ImageFactory& image, vector<shared_ptr<T>>& results)  override;

	private:
		vector< shared_ptr<AbstractFilter <T>> >& h_filters_;
		ThreadPool threadPool_;

	};




}



