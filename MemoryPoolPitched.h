#pragma once

#include <opencv2/highgui/highgui.hpp>
#include <vector>

#include "processing.h"

namespace processing 
{
	using namespace std;
	using namespace cv;

	class MemoryPoolPitched
	{

	public:
		
		MemoryPoolPitched(const MemoryPoolPitched & other) = delete;

		MemoryPoolPitched& operator=(const MemoryPoolPitched & other) = delete;
	
		static MemoryPoolPitched& getMemoryPoolPitchedForOutput();
		
		static MemoryPoolPitched& getMemoryPoolPitchedForInput();
		
		inline vector<float *> getMemory() 
		{
			return memory_;
		}

		inline size_t getPitch() 
		{
			return pitch_;
		}
		
	private:
		MemoryPoolPitched(uint memorySize);
		~MemoryPoolPitched();
		size_t pitch_;
		vector<float *> memory_;

	};

}
