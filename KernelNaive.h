#pragma once
#include "Runnable.h"
#include <vector>
#include "Filter.h"
#include "processing.h"
#include <memory>

using namespace std;
namespace processing
{


	class KernelNaive : public Runnable
	{

	public:

		KernelNaive();

		DELETECOPYASSINGMENT(KernelNaive)

		virtual void run(ImageFactory& image, vector<shared_ptr<Filter>>& filters, vector<shared_ptr<float>>& results) override;

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



