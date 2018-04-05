#pragma once

#include <cuda_runtime.h>

class GpuTimer
{
public:

	GpuTimer()
	{
		cudaEventCreate(&start_);
		cudaEventCreate(&stop_);
	}

	~GpuTimer()
	{
		cudaEventDestroy(start_);
		cudaEventDestroy(stop_);
	}

	void start()
	{
		cudaEventRecord(start_, 0);
	}

	void stop()
	{
		cudaEventRecord(stop_, 0);
	}

	float getElapsedTime()
	{
		float elapsed;
		cudaEventSynchronize(stop_);
		cudaEventElapsedTime(&elapsed, start_, stop_);
		return elapsed;
	}

private:
	cudaEvent_t start_;
	cudaEvent_t stop_;

};
