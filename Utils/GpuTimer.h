#pragma once

#include <cuda_runtime.h>

class GpuTimer
{
public:

	

	GpuTimer(const GpuTimer& other) = delete;

	const GpuTimer& operator=(const GpuTimer& other) = delete;

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

	static GpuTimer& getGpuTimer() 
	{
		static GpuTimer timer;
		return timer;
	}

private:
	GpuTimer()
	{
		cudaEventCreate(&start_);
		cudaEventCreate(&stop_);
	}

	cudaEvent_t start_;
	cudaEvent_t stop_;

};
