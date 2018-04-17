#include "Factory.h"

namespace processing 
{
	Factory::TypeOfConvolution Factory::fromStringToConvolutionType(string text)
	{
		static map<string, Factory::TypeOfConvolution> map
		(
		{
			{ "simple", Factory::TypeOfConvolution::SIMPLE_CONVOLUTION },
			{ "multi", Factory::TypeOfConvolution::MULTI_CONVOLUTION }
		}
		);
		std::transform(text.begin(), text.end(), text.begin(), ::tolower);
		text.erase(remove(text.begin(), text.end(), ' '), text.end());
		auto result = map.find(text);
		return result != map.end() ? result->second : Factory::TypeOfConvolution::NONE;
	}

	Factory::SimpleConvolution Factory::fromStringToSimpleRunnableType(string text)
	{
		static map<string, Factory::SimpleConvolution> map
		(
		{
			{ "cpu", Factory::SimpleConvolution::CPU },
			{ "cpu cropped", Factory::SimpleConvolution::CPU_CROPPED },
			{ "kernel naive", Factory::SimpleConvolution::KERNEL_NAIVE },
			{ "kernel naive improved",Factory::SimpleConvolution::KERNEL_NAIVE_IMPROVED },
			{ "kernel shared full",Factory::SimpleConvolution::KERNEL_SHARED_FULLBLOCK },
			{ "kernel shared incomplete", Factory::SimpleConvolution::KERNEL_SHARED_INCOMPLETEBLOCK },
			{ "kernel shared threads", Factory::SimpleConvolution::KERNEL_SHARED_THREADS }
		}
		);
		std::transform(text.begin(), text.end(), text.begin(), ::tolower);
		trim(text);
		auto result = map.find(text);
		return result != map.end() ? result->second : Factory::SimpleConvolution::NONE;
	}

	Factory::MultiConvolution Factory::fromStringToMultiRunnableType(string text)
	{
		static map<string, Factory::MultiConvolution> map
		(
		{
			{ "cpu", Factory::MultiConvolution::CPU },
			{ "kernel shared multi",Factory::MultiConvolution::KERNEL_SHARED_MULTI },
			{ "kernel shared threads multi", Factory::MultiConvolution::KERNEL_SHARED_THREADS_MULTI }
		}
		);
		std::transform(text.begin(), text.end(), text.begin(), ::tolower);
		trim(text);
		auto result = map.find(text);
		return result != map.end() ? result->second : Factory::MultiConvolution::NONE;
	}

	Factory::TestType Factory::fromStringToTestType(string text)
	{
		static map<string, Factory::TestType> map
		(
		{
			{ "alone", Factory::TestType::ALONE },
			{ "against", Factory::TestType::AGAINST }
		}
		);
		std::transform(text.begin(), text.end(), text.begin(), ::tolower);
		text.erase(remove(text.begin(), text.end(), ' '), text.end());
		auto result = map.find(text);
		return result != map.end() ? result->second : Factory::TestType::NONE;
	}

	Factory::TemplateFactory<Factory::SimpleConvolution>& Factory::getSimpleConvolutionFactory()
	{
		static TemplateFactory<Factory::SimpleConvolution> factory
		(
		{
			{ Factory::SimpleConvolution::CPU, make_shared<CpuSlowConvolution>() },
			{ Factory::SimpleConvolution::CPU_CROPPED, make_shared<CpuCropped>() },
			{ Factory::SimpleConvolution::KERNEL_NAIVE, make_shared<KernelNaive>() },
			{ Factory::SimpleConvolution::KERNEL_NAIVE_IMPROVED, make_shared<KernelNaiveImproved>() },
			{ Factory::SimpleConvolution::KERNEL_SHARED_FULLBLOCK, make_shared<KernelSharedMemoryFullBlock>() },
			{ Factory::SimpleConvolution::KERNEL_SHARED_INCOMPLETEBLOCK, make_shared<KernelSharedMemoryIncompleteBlock>() },
			{ Factory::SimpleConvolution::KERNEL_SHARED_THREADS, make_shared<KernelSharedThreads>() }
		}
		);
		return factory;
	}

	Factory::TemplateFactory<Factory::MultiConvolution>& Factory::getMultiConvolutionFactory()
	{
		static TemplateFactory<Factory::MultiConvolution> factory
		(
		{
			{ Factory::MultiConvolution::CPU, make_shared<CpuCroppedMulti>() },
			{ Factory::MultiConvolution::KERNEL_SHARED_MULTI, make_shared<KernelSharedMulti>() },
			{ Factory::MultiConvolution::KERNEL_SHARED_THREADS_MULTI, make_shared<KernelSharedThreadsMulti>() },
		}
		);
		return factory;
	}


}


