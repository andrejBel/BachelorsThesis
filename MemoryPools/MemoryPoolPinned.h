#pragma once
#include <mutex>
#include <condition_variable>
#include <opencv2/highgui/highgui.hpp>
#include <stack>
#include <memory>
#include "processing.h"
namespace processing 
{
	using namespace std;
	using namespace cv;

	class MemoryPoolPinned
	{

	public:
		
		~MemoryPoolPinned();
		
		MemoryPoolPinned(const MemoryPoolPinned & other) = delete;

		MemoryPoolPinned& operator=(const MemoryPoolPinned & other) = delete;

		shared_ptr<float> acquireMemory(const size_t size, const bool preferPinned);
	
		static MemoryPoolPinned& getMemoryPoolPinnedForOutput();
		
		static MemoryPoolPinned& getMemoryPoolPinnedForInput();
		
		inline auto  getBufferSize() 
		{
			return buffer_.size();
		}


	private:
		MemoryPoolPinned(uint bufferSize);
		void releaseMemory(float* memory);

		stack<float *>  buffer_;
		mutex mutex_;
		condition_variable conditionVariable_;


	};

}




