#pragma once
#include <opencv2/highgui/highgui.hpp>
#include <stack>
#include <memory>
#include "processing.h"
namespace processing
{
	using namespace std;
	using namespace cv;

	class MemoryPoolManaged
	{

	public:

		~MemoryPoolManaged();

		MemoryPoolManaged(const MemoryPoolManaged & other) = delete;

		MemoryPoolManaged& operator=(const MemoryPoolManaged & other) = delete;

		shared_ptr<float> acquireMemory(const size_t size);

		static MemoryPoolManaged& getMemoryPoolManaged();

	private:
		MemoryPoolManaged(uint bufferSize);
		void releaseMemory(float* memory);

		stack<float *>  buffer_;
		mutex mutex_;
		condition_variable conditionVariable_;


	};

}