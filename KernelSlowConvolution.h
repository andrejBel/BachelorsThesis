#pragma once
#include "Runnable.h"
#include <vector>
#include "Filter.h"
#include <memory>

using namespace std;
namespace processing 
{

	template<typename T>
	class KernelSlowConvolution : public Runnable<T>
	{
		static_assert(std::is_floating_point<T>::value, "Class KernelSlowConvolution can only be instantiazed with float, double or long double");
	public:

		KernelSlowConvolution(vector< shared_ptr<AbstractFilter <T>> >& filters);

		DELETECOPYASSINGMENT(KernelSlowConvolution<T>)

			virtual void run(ImageFactory& image, vector<shared_ptr<T>>& results)  override;

	private:
		vector< shared_ptr<AbstractFilter <T>> >& h_filters_;

	};

	


}



