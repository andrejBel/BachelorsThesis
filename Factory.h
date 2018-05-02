#pragma once

#include <map>
#include <memory>
#include <algorithm>
#include <string>
#include "CpuCropped.h"
#include "CpuCroppedMulti.h"
#include "CpuSlowConvolution.h"

#include "KernelNaive.h"
#include "KernelNaiveImproved.h"
#include "KernelSharedMemoryFullBlock.h"
#include "KernelSharedMemoryIncompleteBlock.h"
#include "KernelSharedMulti.h"
#include "KernelSharedThreads.h"
#include "KernelSharedThreadsMulti.h"

using namespace std;

namespace processing
{




	class Factory
	{
	private:

		template <typename ENUM_TYPE>
		class TemplateFactory
		{
		public:
			TemplateFactory(map<ENUM_TYPE, shared_ptr<Runnable>> map) : factoryMap_(map)
			{}
			

			~TemplateFactory() {};

			TemplateFactory(const TemplateFactory& other) = delete;
			TemplateFactory& operator=(const TemplateFactory& other) = delete;

			shared_ptr<Runnable> getRunnable(ENUM_TYPE type)
			{
				auto find = factoryMap_.find(type);
				return find != factoryMap_.end() ? find->second : shared_ptr<Runnable>(nullptr);
			}

		private:
			map<ENUM_TYPE, shared_ptr<Runnable>> factoryMap_;

		};

		Factory()
		{}

		static std::string& ltrim(std::string& s, const char* t = " \t\n\r\f\v")
		{
			s.erase(0, s.find_first_not_of(t));
			return s;
		}

		static std::string& rtrim(std::string& s, const char* t = " \t\n\r\f\v")
		{
			s.erase(s.find_last_not_of(t) + 1);
			return s;
		}

		static std::string& trim(std::string& s, const char* t = " \t\n\r\f\v")
		{
			return ltrim(rtrim(s, t), t);
		}


	public:

		enum class TypeOfConvolution
		{
			SIMPLE_CONVOLUTION = 0,
			MULTI_CONVOLUTION = 1,
			NONE = 2
		};

		enum class SimpleConvolution
		{
			CPU = 0,
			CPU_CROPPED = 1,
			KERNEL_NAIVE = 2,
			KERNEL_NAIVE_IMPROVED = 3,
			KERNEL_SHARED_FULLBLOCK = 4,
			KERNEL_SHARED_INCOMPLETEBLOCK = 5,
			KERNEL_SHARED_THREADS = 6,
			NONE = 7
		};

		enum class MultiConvolution
		{
			CPU = 0,
			KERNEL_SHARED_MULTI = 1,
			KERNEL_SHARED_THREADS_MULTI = 2,
			NONE = 3
		};

		enum class TestType
		{
			ALONE = 0,
			AGAINST = 1,
			NONE = 2
		};

		enum class OutputType
		{
			IMAGE = 0,
			TEXTFILE = 1,
			NONE = 2
		};


		static Factory::TypeOfConvolution fromStringToConvolutionType(string text);

		static Factory::SimpleConvolution fromStringToSimpleRunnableType(string text);
		
		static Factory::MultiConvolution fromStringToMultiRunnableType(string text);
		
		static Factory::TestType fromStringToTestType(string text);

		static Factory::OutputType fromStringToOutputType(string text);
		

		Factory(const Factory& other) = delete;

		Factory& operator=(const Factory& other) = delete;

		static Factory::TemplateFactory<Factory::SimpleConvolution>& getSimpleConvolutionFactory();

		static Factory::TemplateFactory<Factory::MultiConvolution>& getMultiConvolutionFactory();

	};

}



