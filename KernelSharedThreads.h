#pragma once
#include "Runnable.h"
#include <vector>
#include "processing.h"
#include "Filter.h"
#include <memory>
#include <string>

using namespace std;
namespace processing
{


	class KernelSharedThreads : public Runnable
	{

	public:

		KernelSharedThreads();

		DELETECOPYASSINGMENT(KernelSharedThreads)

		virtual void run(ImageFactory& image, vector<shared_ptr<AbstractFilter>>& filters, vector<shared_ptr<float>>& results)  override;

		virtual string getDescription() override
		{
			return "GPU shared memmory for small";
		}

	private:

	};

	struct CudaStream
	{
		CudaStream()
		{
			cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
		}

		~CudaStream()
		{
			checkCudaErrors(cudaStreamDestroy(stream_));
		}

		cudaStream_t stream_;

	};

}



