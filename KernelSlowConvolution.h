#pragma once
#include "Runnable.h"
#include <vector>
#include "Filter.h"

using namespace std;
namespace processing 
{

	template<typename T>
	class KernelSlowConvolution : public Runnable
	{
		static_assert(std::is_floating_point<T>::value, "Class KernelSlowConvolution can only be instantiazed with float, double or long double");
	public:

		KernelSlowConvolution(vector<Filter<T>>& filters);

		DELETECOPYASSINGMENT(KernelSlowConvolution)

		virtual void run(ImageFactory & image) override;

	private:
		vector<Filter<T>>& h_filters_;

	};

	


}



