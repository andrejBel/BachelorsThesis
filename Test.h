#pragma once
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <algorithm>
#include "opencv2/core/utility.hpp"
#include "Filter.h"
#include "Runnable.h"
#include "processing.h"
#include "KernelSharedMemory.h"
#include "KernelSlowConvolution.h"
#include "KernelSlowConvolutionNoEdgeCopy.h"
#include "CpuSlowConvolution.h"
#include "CPUSlowConvolutionAsync.h"
#include "KernelSharedMemoryAsync.h"

using namespace std;
using namespace cv;

namespace processing 
{
	static const string INPUT_IMAGE_PATH = "input_img.jpg";

	template <typename T>
	class Test
	{
	public:


		Test(): 
		fileName_(INPUT_IMAGE_PATH),
		epsilon_(0.001),
		replications_(1)
		{}

		vector<vector<shared_ptr<T>>> operator()();
		vector<vector<shared_ptr<T>>> testForMannaged();
		

		static void testAlone(shared_ptr<Runnable<T>> runnable, uint replications = 1);
		static void testAloneForManaged(shared_ptr<Runnable<T>> runnable, uint replications = 1);
		static void testAgainstCpuMulticore(shared_ptr<Runnable<T>> runnable, uint replications = 1);
		static void testAgainstCpuSingleCore(shared_ptr<Runnable<T>> runnable, uint replications = 1);


		static void testAllAgainstCpu();

		class Builder
		{
		public:
			Builder& addFilter(shared_ptr<AbstractFilter<T>> filter) 
			{
				test_.filters_.push_back(filter);
				return *this;
			}

			Builder& setFilters(vector<shared_ptr<AbstractFilter<T>>> filters)
			{
				test_.filters_ = filters;
				return *this;
			}

			Builder& addRunnable(shared_ptr<Runnable<T>> runnable)
			{
				test_.runnables_.push_back(runnable);
				return *this;
			}

			Builder& setRunnables(vector<shared_ptr<Runnable<T>>> runnables)
			{
				test_.runnables_ = runnables;
				return *this;
			}

			Builder& setReplications(uint replications)
			{
				test_.replications_ = replications;
				return *this;
			}

			Builder& setImagePath(const string& path)
			{
				test_.fileName_ = path;
				return *this;
			}

			Builder& setEpsilon(T epsilon)
			{
				test_.epsilon_ = epsilon;
				return *this;
			}

			Test build() 
			{
				return test_;
			}

		private:
			Test test_;
		};

		

	private:
		vector<shared_ptr<AbstractFilter<T>>> filters_;
		vector<shared_ptr<Runnable<T>>> runnables_;
		uint replications_;
		string fileName_;
		T epsilon_;

		static shared_ptr<AbstractFilter<T>> get3x3Filter() 
		{
			static shared_ptr<AbstractFilter<T>> filter = createFilter<T>
				(3,
					{
					-10.0f,8.4f,11.0f,
					2.0f,4.3f,22.0f,
					-1.0f,2.5f,12.0f
					}, 1.0f
				);
			return filter;
		}

		static shared_ptr<AbstractFilter<T>> get5x5Filter() 
		{
			static shared_ptr<AbstractFilter<T>> filter = createFilter<T>
				(5,
					{
					-1.0f,2.0f,1.0f,1.5f, 1.6f,
					-2.0f,7.0f,2.0f,5.8f,12.1f,
					-1.0f,8.0f,2.0f,4.7f,3.3f
					}, 1.7f
				);
			return filter;
		}

		static shared_ptr<AbstractFilter<T>> get7x7Filter()
		{
			static shared_ptr<AbstractFilter<T>> filter = createFilter<T>
				(7, 
					{
					0.00000067f,0.00002292f,0.00019117f,0.00038771f,	0.00019117f,	0.00002292f,	0.00000067f,
					0.00002292f,	0.00078634f,	0.00655965f,	0.01330373f,	0.00655965f,	0.00078633f,	0.00002292f,
					0.00019117f,	0.00655965f,	0.05472157f,	0.11098164f,	0.05472157f,	0.00655965f,	0.00019117f,
					0.00038771f,	0.01330373f,	0.11098164f,	0.22508352f,	0.11098164f,	0.01330373f,	0.00038771f,
					0.00019117f,	0.00655965f,	0.05472157f,	0.11098164f,	0.05472157f,	0.00655965f,	0.00019117f,
					0.00002292f,	0.00078633f,	0.00655965f,	0.01330373f,	0.00655965f,	0.00078633f, 0.00002292f,
					0.00000067f,	0.00002292f,	0.00019117f,	0.00038771f,	0.00019117f,	0.00002292f,	0.00000067f
					}, 2.0f
				);
			return filter;
		}

	};




	template<typename T>
	inline vector<vector<shared_ptr<T>>> Test<T>::operator()()
	{
		vector<vector<shared_ptr<T>>> results;
		ImageFactory image(fileName_);
		TickMeter meter;
		for (shared_ptr<Runnable<T>> & runnable : runnables_)
		{
			double time(0);
			results.push_back(vector<shared_ptr<T>>());
			for (uint i = 0; i < replications_; i++)
			{
				meter.reset();
				meter.start();
				runnable->run(image, filters_, results[results.size() - 1]);
				meter.stop();
				time += meter.getTimeMilli();
			}
			
			cout << runnable->getDescription() << ". Replications:  " << replications_ << ". Average time: " << (time / replications_) << endl;
		}
		
		if (runnables_.size() == 2) 
		{
			vector<shared_ptr<T>>& resultsFirst = results[0];
			vector<shared_ptr<T>>& resultsSecond = results[1];
			size_t pixels = image.getNumPixels();
			auto size = std::min(resultsFirst.size(), resultsSecond.size());
			for (size_t i = 0; i < size; i++)
			{
				shared_ptr<T> resultFirst = resultsFirst[i];
				shared_ptr<T> resultSecond = resultsSecond[i];
				for (size_t j = 0; j < pixels; j++)
				{
					if ([&resultFirst, &resultSecond, j]() {
						return fabs(resultFirst.get()[j] - resultSecond.get()[j]) > 0.1;
					}())
					{
						cout << "-----------------------" << endl;
						cout << "Index: " << j << ", epsilon: " << epsilon_ << endl;
						cout << runnables_[0]->getDescription() << ": " << resultFirst.get()[j] << endl;
						cout << runnables_[1]->getDescription() << ": " << resultSecond.get()[j] << endl;
						cout << runnables_[0]->getDescription() << " - " << runnables_[1]->getDescription() << ": " << resultFirst.get()[j] - resultSecond.get()[j] << endl;
						cout << "-----------------------" << endl;
					}
				}
			}
		}
		return results;
	}

	template<typename T>
	inline vector<vector<shared_ptr<T>>> Test<T>::testForMannaged()
	{
		vector<vector<shared_ptr<T>>> results;
		ImageFactory image(fileName_);
		TickMeter meter;
		for (shared_ptr<Runnable<T>> & runnable : runnables_)
		{
			double time(0);
			results.push_back(vector<shared_ptr<T>>());
			for (uint i = 0; i < replications_; i++)
			{
				meter.reset();
				size_t pixels = image.getNumPixels();
				auto& result = results[results.size() - 1];
				meter.start();
				runnable->run(image, filters_, result);
				size_t size = result.size();
				for (size_t j = 0; j < size; j++)
				{
					T * memory = result[0].get();
					T temp;
					for (size_t k = 0; k < pixels; k++)
					{
						temp = memory[j];
					}
				}
				meter.stop();
				time += meter.getTimeMilli();
			}
			cout << runnable->getDescription() << ". Replications:  " << replications_ << ". Average time: " << (time / replications_) << endl;
		}
		return results;
	}

	template<typename T>
	inline void Test<T>::testAlone(shared_ptr<Runnable<T>> runnable, uint replications)
	{
		Test<T>::Builder builder;
		builder.setFilters({ Test<T>::get3x3Filter() , Test<T>::get5x5Filter(), Test<T>::get7x7Filter(), Test<T>::get3x3Filter() , Test<T>::get5x5Filter(), Test<T>::get7x7Filter(),Test<T>::get7x7Filter(),Test<T>::get7x7Filter(),Test<T>::get7x7Filter(),Test<T>::get7x7Filter(),Test<T>::get7x7Filter(),Test<T>::get7x7Filter(),Test<T>::get7x7Filter(),Test<T>::get7x7Filter(),Test<T>::get7x7Filter(),Test<T>::get7x7Filter(),Test<T>::get7x7Filter(),Test<T>::get7x7Filter(),Test<T>::get7x7Filter(),Test<T>::get7x7Filter(),Test<T>::get7x7Filter() }).addRunnable(runnable).setReplications(replications);
		builder.build()();
	}

	template<typename T>
	inline void Test<T>::testAloneForManaged(shared_ptr<Runnable<T>> runnable, uint replications)
	{
		Test<T>::Builder builder;
		builder.setFilters({ Test<T>::get3x3Filter() , Test<T>::get5x5Filter(), Test<T>::get7x7Filter() }).addRunnable(runnable).setReplications(replications);
		builder.build().testForMannaged();
	}

	template<typename T>
	inline void Test<T>::testAgainstCpuMulticore(shared_ptr<Runnable<T>> runnable, uint replications)
	{
		Test<T>::Builder builder;
		builder.setFilters({ Test<T>::get3x3Filter() , Test<T>::get5x5Filter(), Test<T>::get7x7Filter() }).addRunnable(runnable).addRunnable( make_shared<CPUSlowConvolutionAsync<T>>()).setReplications(replications);
		builder.build()();
	}

	template<typename T>
	inline void Test<T>::testAgainstCpuSingleCore(shared_ptr<Runnable<T>> runnable, uint replications)
	{
		Test<T>::Builder builder;
		builder.setFilters({ Test<T>::get3x3Filter() , Test<T>::get5x5Filter(), Test<T>::get7x7Filter() }).addRunnable(runnable).addRunnable(make_shared<CpuSlowConvolution<T>>()).setReplications(replications);
		builder.build()();
	}

	template<typename T>
	inline void Test<T>::testAllAgainstCpu()
	{
		Test<T>::Builder builder;
		vector<shared_ptr<AbstractFilter<T>>> filters = { Test<T>::get3x3Filter() , Test<T>::get5x5Filter(), Test<T>::get7x7Filter() };
		builder.setFilters(filters);
		{
			vector<shared_ptr<Runnable<T>>> runnables = { make_shared<KernelSharedMemory<T>>(), make_shared<CpuSlowConvolution<T>>() };
			builder.setRunnables(runnables);
			builder.build()();
		}
		{
			vector<shared_ptr<Runnable<T>>> runnables = { make_shared<KernelSlowConvolution<T>>(), make_shared<CpuSlowConvolution<T>>() };
			builder.setRunnables(runnables);
			builder.build()();
		}
		{
			vector<shared_ptr<Runnable<T>>> runnables = { make_shared<KernelSlowConvolutionNoEdgeCopy<T>>(), make_shared<CpuSlowConvolution<T>>() };
			builder.setRunnables(runnables);
			builder.build()();
		}
		{
			vector<shared_ptr<Runnable<T>>> runnables = { make_shared<CPUSlowConvolutionAsync<T>>(), make_shared<CpuSlowConvolution<T>>() };
			builder.setRunnables(runnables);
			builder.build()();
		}

	}

}


