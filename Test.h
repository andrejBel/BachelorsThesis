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
#include "KernelSharedMemoryManaged.h"

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
	
	public:
		//excel generate command
		//=CONCAT(SUBSTITUTE(TEXT(ROUND(RAND()*RAND()*0,4;6);"0,000000");",";".";1);"f,")
		static shared_ptr<AbstractFilter<T>> get1x1Filter()
		{
			static shared_ptr<AbstractFilter<T>> filter = createFilter<T>
				(1,
					{
					2.8f
					}, 1.8f
				);
			return filter;
		}

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
					-1.0f,8.0f,2.0f,4.7f,3.3f,
					-4.1f,1.28f,0.14f,2.3f,0.38f,
					-1.0f,4.56f,10.02f,4.73f,3.33f
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

		static shared_ptr<AbstractFilter<T>> get9x9Filter()
		{
			static shared_ptr<AbstractFilter<T>> filter = createFilter<T>
				(9,
				{
					0.036278f,	0.089869f,	0.001261f,	0.009317f,	0.151778f,	0.107585f,	0.182440f,	0.042363f,	0.000292f,
					0.035093f,	0.002465f,	0.106844f,	0.126285f,	0.117432f,	0.071919f,	0.101333f,	0.089229f,	0.293655f,
					0.107707f,	0.060222f,	0.095177f,	0.002516f,	0.005926f,	0.019379f,	0.005608f,	0.077404f,	0.026859f,
					0.129057f,	0.224715f,	0.054397f,	0.027630f,	0.200282f,	0.054348f,	0.067945f,	0.312950f,	0.108934f,
					0.193089f,	0.063988f,	0.323055f,	0.029709f,	0.067582f,	0.091858f,	0.132132f,	0.038692f,	0.216645f,
					0.022046f,	0.009278f,	0.071635f,	0.132760f,	0.003295f,	0.048065f,	0.002239f,	0.066107f,	0.349744f,
					0.115120f,	0.113435f,	0.035781f,	0.193100f,	0.086339f,	0.067375f,	0.058935f,	0.061868f,	0.053128f,
					0.001438f,	0.265224f,	0.017876f,	0.062295f,	0.092051f,	0.289665f,	0.266714f,	0.052215f,	0.035599f,
					0.177824f,	0.013773f,	0.067462f,	0.298867f,	0.014095f,	0.120466f,	0.091443f,	0.093035f,	0.116388f

				}, 0.5f
					);
			return filter;
		}

		static shared_ptr<AbstractFilter<T>> get11x11Filter()
		{
			static shared_ptr<AbstractFilter<T>> filter = createFilter<T>
				(11,
				{
					0.006532f,	0.110691f,	0.184945f,	0.021727f,	0.025837f,	0.033302f,	0.081124f,	0.127049f,	0.182580f,	0.217225f,	0.048565f,
					0.119473f,	0.187317f,	0.065881f,	0.019231f,	0.305912f,	0.067432f,	0.053566f,	0.338248f,	0.041688f,	0.297081f,	0.111158f,
					0.007027f,	0.024264f,	0.020725f,	0.117846f,	0.047676f,	0.006651f,	0.113976f,	0.062822f,	0.009223f,	0.058222f,	0.016112f,
					0.054365f,	0.078013f,	0.012604f,	0.017742f,	0.100898f,	0.176327f,	0.263560f,	0.078225f,	0.000288f,	0.025116f,	0.128227f,
					0.052624f,	0.049526f,	0.145277f,	0.135679f,	0.003016f,	0.104771f,	0.165465f,	0.053195f,	0.376627f,	0.018251f,	0.077947f,
					0.083925f,	0.125662f,	0.021535f,	0.256193f,	0.100542f,	0.354919f,	0.013374f,	0.221924f,	0.199758f,	0.173118f,	0.147557f,
					0.012797f,	0.352680f,	0.095941f,	0.006091f,	0.041444f,	0.038900f,	0.000106f,	0.167763f,	0.085543f,	0.137201f,	0.126702f,
					0.048753f,	0.117384f,	0.029390f,	0.084432f,	0.046492f,	0.091720f,	0.054221f,	0.042305f,	0.155931f,	0.094010f,	0.117046f,
					0.012171f,	0.041480f,	0.142677f,	0.001661f,	0.090166f,	0.153569f,	0.050737f,	0.183825f,	0.265724f,	0.005598f,	0.000512f,
					0.167192f,	0.068927f,	0.000780f,	0.078211f,	0.223493f,	0.086666f,	0.011663f,	0.038719f,	0.134682f,	0.078469f,	0.356147f,
					0.005099f,	0.223795f,	0.207450f,	0.028848f,	0.207032f,	0.122272f,	0.251497f,	0.159479f,	0.017581f,	0.021751f,	0.020274f
				}, 1.5f
					);
			return filter;
		}

		static shared_ptr<AbstractFilter<T>> get13x13Filter()
		{
			static shared_ptr<AbstractFilter<T>> filter = createFilter<T>
				(13,
				{
					0.056964f,	0.003826f,	0.198016f,	0.024934f,	0.020462f,	0.285452f,	0.129167f,	0.050595f,	0.285314f,	0.005765f,	0.051772f,	0.024044f,	0.079132f,
					0.221735f,	0.118979f,	0.002893f,	0.180767f,	0.063807f,	0.267696f,	0.062351f,	0.024466f,	0.306390f,	0.212091f,	0.159483f,	0.141597f,	0.012792f,
					0.096225f,	0.219310f,	0.005326f,	0.102385f,	0.063288f,	0.004076f,	0.154125f,	0.146647f,	0.004960f,	0.115788f,	0.204344f,	0.013670f,	0.236634f,
					0.065037f,	0.166449f,	0.097978f,	0.093337f,	0.131066f,	0.099354f,	0.099893f,	0.009026f,	0.258459f,	0.021879f,	0.009342f,	0.040802f,	0.111769f,
					0.100039f,	0.044386f,	0.087018f,	0.007485f,	0.040909f,	0.130834f,	0.154452f,	0.122095f,	0.084909f,	0.066843f,	0.156487f,	0.003058f,	0.125446f,
					0.063283f,	0.061510f,	0.232438f,	0.196814f,	0.214310f,	0.002815f,	0.130219f,	0.101929f,	0.286077f,	0.202701f,	0.048566f,	0.037916f,	0.017847f,
					0.233739f,	0.003706f,	0.153824f,	0.052712f,	0.058516f,	0.071055f,	0.036971f,	0.096532f,	0.038364f,	0.031370f,	0.124757f,	0.196667f,	0.055228f,
					0.040803f,	0.028000f,	0.186267f,	0.042664f,	0.112157f,	0.131404f,	0.103408f,	0.048206f,	0.001395f,	0.317284f,	0.013802f,	0.016314f,	0.269215f,
					0.040776f,	0.020588f,	0.011222f,	0.056144f,	0.065447f,	0.298003f,	0.206688f,	0.120290f,	0.023319f,	0.016045f,	0.046136f,	0.321421f,	0.013347f,
					0.067075f,	0.154030f,	0.097136f,	0.191440f,	0.057317f,	0.108293f,	0.031817f,	0.066330f,	0.005386f,	0.022877f,	0.017486f,	0.055357f,	0.034743f,
					0.062356f,	0.061261f,	0.231216f,	0.057266f,	0.110784f,	0.010923f,	0.132973f,	0.045575f,	0.074787f,	0.075412f,	0.007030f,	0.016349f,	0.028186f,
					0.044224f,	0.003224f,	0.026080f,	0.067239f,	0.205584f,	0.204141f,	0.112010f,	0.169499f,	0.147127f,	0.071066f,	0.024834f,	0.271579f,	0.141831f,
					0.095209f,	0.153787f,	0.011107f,	0.024648f,	0.043237f,	0.328715f,	0.001800f,	0.001390f,	0.034060f,	0.123419f,	0.072912f,	0.040718f,	0.204140f
				}, 1.0f
				);
			return filter;
		}

		static shared_ptr<AbstractFilter<T>> get15x15Filter()
		{
			static shared_ptr<AbstractFilter<T>> filter = createFilter<T>
				(15,
				{
					0.082687f,	0.026221f,	0.052721f,	0.077936f,	0.067556f,	0.059633f,	0.037684f,	0.029745f,	0.268608f,	0.222160f,	0.013158f,	0.002824f,	0.009861f,	0.004119f,	0.118977f,
					0.042203f,	0.000232f,	0.012068f,	0.030282f,	0.038332f,	0.034134f,	0.087371f,	0.083201f,	0.141348f,	0.099946f,	0.072423f,	0.145292f,	0.138904f,	0.196859f,	0.359142f,
					0.033710f,	0.092225f,	0.156470f,	0.016185f,	0.142548f,	0.016067f,	0.023921f,	0.046760f,	0.020434f,	0.013818f,	0.049233f,	0.012222f,	0.069602f,	0.004977f,	0.123298f,
					0.012481f,	0.074636f,	0.025311f,	0.014024f,	0.062068f,	0.225618f,	0.009153f,	0.270775f,	0.060281f,	0.383816f,	0.032751f,	0.089439f,	0.178188f,	0.046354f,	0.180820f,
					0.066983f,	0.002019f,	0.006039f,	0.135296f,	0.045973f,	0.024139f,	0.084502f,	0.022458f,	0.022427f,	0.249530f,	0.001546f,	0.098715f,	0.022289f,	0.003653f,	0.080430f,
					0.100464f,	0.022543f,	0.061206f,	0.053897f,	0.123321f,	0.220146f,	0.027385f,	0.081530f,	0.031628f,	0.001052f,	0.187625f,	0.033054f,	0.019997f,	0.014771f,	0.108846f,
					0.044959f,	0.249404f,	0.153375f,	0.011017f,	0.078456f,	0.358350f,	0.000056f,	0.010079f,	0.087625f,	0.198029f,	0.166266f,	0.034739f,	0.183114f,	0.219950f,	0.009701f,
					0.149279f,	0.064749f,	0.119080f,	0.015548f,	0.059721f,	0.003711f,	0.237501f,	0.048963f,	0.146513f,	0.143810f,	0.324437f,	0.104964f,	0.174161f,	0.016911f,	0.009777f,
					0.085571f,	0.069976f,	0.354110f,	0.005487f,	0.240364f,	0.032664f,	0.192692f,	0.020607f,	0.008016f,	0.085407f,	0.025257f,	0.347013f,	0.008420f,	0.094937f,	0.024716f,
					0.129176f,	0.083138f,	0.000059f,	0.121617f,	0.000359f,	0.117681f,	0.012565f,	0.006107f,	0.112241f,	0.094908f,	0.119760f,	0.322765f,	0.106818f,	0.070734f,	0.363759f,
					0.098018f,	0.014171f,	0.190410f,	0.184023f,	0.339279f,	0.101357f,	0.130484f,	0.007610f,	0.137705f,	0.024271f,	0.103057f,	0.069406f,	0.100261f,	0.153338f,	0.003564f,
					0.115220f,	0.025227f,	0.333314f,	0.173028f,	0.290992f,	0.133896f,	0.043709f,	0.110049f,	0.139719f,	0.068038f,	0.006499f,	0.251913f,	0.115868f,	0.038520f,	0.143770f,
					0.295509f,	0.106248f,	0.003647f,	0.014268f,	0.048997f,	0.029752f,	0.005931f,	0.027171f,	0.009999f,	0.012123f,	0.013846f,	0.047701f,	0.136559f,	0.185118f,	0.093833f,
					0.068555f,	0.083186f,	0.008739f,	0.002388f,	0.112365f,	0.073594f,	0.090579f,	0.073873f,	0.008145f,	0.209170f,	0.171435f,	0.163713f,	0.028474f,	0.047811f,	0.065447f,
					0.036554f,	0.014620f,	0.031659f,	0.000093f,	0.063266f,	0.023471f,	0.141030f,	0.034365f,	0.302973f,	0.278303f,	0.208603f,	0.035180f,	0.079931f,	0.073000f,	0.105001f
				}, 1.2f
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
		builder.setFilters({ Test<T>::get3x3Filter(), Test<T>::get5x5Filter(), Test<T>::get7x7Filter(), Test<T>::get3x3Filter() , Test<T>::get5x5Filter(), Test<T>::get7x7Filter() }).addRunnable(runnable).setReplications(replications);
		builder.build().testForMannaged();
	}

	template<typename T>
	inline void Test<T>::testAgainstCpuMulticore(shared_ptr<Runnable<T>> runnable, uint replications)
	{
		Test<T>::Builder builder;
		builder
		//.addFilter(Test<T>::get1x1Filter())
		.addFilter(Test<T>::get3x3Filter())
		.addFilter(Test<T>::get5x5Filter())
		//.addFilter(Test<T>::get7x7Filter())
		//.addFilter(Test<T>::get9x9Filter())
		//.addFilter(Test<T>::get11x11Filter())
		//.addFilter(Test<T>::get13x13Filter())
		//.addFilter(Test<T>::get15x15Filter())
		.addRunnable(runnable).addRunnable( make_shared<CPUSlowConvolutionAsync<T>>()).setReplications(replications);
		builder.build()();
	}

	template<typename T>
	inline void Test<T>::testAgainstCpuSingleCore(shared_ptr<Runnable<T>> runnable, uint replications)
	{
		Test<T>::Builder builder;
		builder
			.addFilter(Test<T>::get1x1Filter())
			.addFilter(Test<T>::get3x3Filter())
			.addFilter(Test<T>::get5x5Filter())
			.addFilter(Test<T>::get7x7Filter())
			.addFilter(Test<T>::get9x9Filter())
			.addFilter(Test<T>::get11x11Filter())
			.addFilter(Test<T>::get13x13Filter())
			.addFilter(Test<T>::get15x15Filter()).addRunnable(runnable).addRunnable(make_shared<CpuSlowConvolution<T>>()).setReplications(replications);
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


