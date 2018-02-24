#pragma once
#include "Runnable.h"
#include <vector>
#include "Filter.h"
#include <memory>

using namespace std;

namespace processing 
{

	template<typename T>
	class CpuSlowConvolution : public Runnable
	{
		static_assert(std::is_floating_point<T>::value, "Class KernelSlowConvolution can only be instantiazed with float, double or long double");
	public:

		CpuSlowConvolution(vector< shared_ptr<AbstractFilter <T>> >& filters);

		DELETECOPYASSINGMENT(CpuSlowConvolution<T>)

		virtual void run(ImageFactory & image) override;

	private:
		vector< shared_ptr<AbstractFilter <T>> >& filters_;

	};

}



