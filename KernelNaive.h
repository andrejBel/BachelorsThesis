#pragma once
#include "Runnable.h"
#include <vector>
#include "Filter.h"
#include <memory>

using namespace std;
namespace processing
{

	template<typename T>
	class KernelNaive : public Runnable<T>
	{
		static_assert(std::is_floating_point<T>::value, "Class KernelSlowConvolution can only be instantiazed with float, double or long double");
	public:

		KernelNaive();

		DELETECOPYASSINGMENT(KernelNaive<T>)

		virtual void run(ImageFactory& image, vector<shared_ptr<AbstractFilter<T>>>& filters, vector<shared_ptr<T>>& results) override;

		virtual string getDescription() override
		{
			return "CPU slow naive implementation";
		}

		template <typename T>
		__device__ static const T& min(const T& a, const T&b)
		{
			return a < b ? a : b;
		}

		template <typename T>
		__device__ static const T& max(const T& a, const T&b)
		{
			return a > b ? a : b;
		}

	private:


	};




}



