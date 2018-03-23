#pragma once

#include "device_launch_parameters.h"

#include <cuda_runtime.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/core/utility.hpp"
#include <opencv2/opencv.hpp>

#include <string>
#include <memory>
#include <vector>
#include "Filter.h"
#include <mutex>
#include <condition_variable>

using namespace std;
using namespace cv;



namespace processing 
{

	template <typename int N>
	struct Box
	{
		CPUGPU Box() {}

		CPUGPU Box(Box& other) {
			for (int i = 0; i < N; i++)
			{
				this->memory_[i] = other.memory_[i];
			}
		}

		float * memory_[N];

	};

	template <typename int N>
	struct QueueBuffer
	{
	public:

		void printBuf(int index, int howMuch)
		{
			for (int i = 0; i < howMuch; i++)
			{
				cout << i << ": " << memory_[(index + i) % N] << endl;
			}
		}

		// return index of beginning 
		int acquire(int requirement = 1)
		{
			if (requirement > N) {
				throw std::runtime_error("Requirment bigger than buffer");
			}
			unique_lock<mutex> lock(mutex_);
			while (requirement > (N - used_))
			{
				conditionVariable_.wait(lock);
			}
			int index = (start_ + used_) % N;
			used_ += requirement;
			return index;
		}

		//size of returned source
		void release(int inReturn)
		{
			unique_lock<mutex> lock(mutex_);
			if (inReturn > used_)
			{
				throw std::runtime_error("Requirment bigger than buffer");
			}
			start_ += inReturn;
			start_ %= N;
			used_ -= inReturn;
			lock.unlock();
			conditionVariable_.notify_one();
		}

		mutex mutex_;
		condition_variable conditionVariable_;

		int start_ = 0;
		int used_ = 0;
		float * memory_[N];

	};

	static const uint MAX_IMAGE_WIDTH = 2000;
	static const uint MAX_IMAGE_HEIGHT = 2000;
	static const size_t MAX_IMAGE_RESOLUTION = MAX_IMAGE_WIDTH*MAX_IMAGE_HEIGHT;
	static const int PINNED_MEMORY_BUFFER_SIZE_INPUT = 5;
	static const int PINNED_MEMORY_BUFFER_SIZE_OUTPUT = 40;
	static const int PITCHED_MEMORY_BUFFER_SIZE_INPUT = 2;
	static const int PITCHED_MEMORY_BUFFER_SIZE_OUTPUT = 10;

	const int MAXFILTERWIDTH = 15;
	static __constant__ float FILTERCUDA[MAXFILTERWIDTH * MAXFILTERWIDTH * PITCHED_MEMORY_BUFFER_SIZE_OUTPUT];
	static __constant__ Box<PITCHED_MEMORY_BUFFER_SIZE_OUTPUT> PITCHED_MEMORY_BUFFER_DEVICE; 
	static __device__ __constant__ size_t INPUT_PITCH_DEVICE[1];
	static __device__ __constant__ size_t OUTPUT_PITCH_DEVICE[1];

	static QueueBuffer<PITCHED_MEMORY_BUFFER_SIZE_OUTPUT> PITCHED_MEMORY_BUFFER_HOST;
	

	class Runnable;

	class ImageFactory
	{
	public:
		ImageFactory(const string& fileName);

		ImageFactory(const ImageFactory& other) = delete;

		ImageFactory& operator=(const ImageFactory& other) = delete;
	

		inline uchar4* getInputRGBAPointer() 
		{
			return reinterpret_cast<uchar4 *>(imageRGBAInput_.ptr<uchar>(0));
		}

		inline uchar* getInputGrayPointer()
		{
			return imageGrayInput_.ptr<uchar>(0);
		}

		inline uchar4* getOutputRGBAPointer()
		{
			return reinterpret_cast<uchar4 *>(imageRGBAOutput_.ptr<uchar>(0));
		}

		inline uchar* getOutputGrayPointer()
		{
			return imageGrayOutput_.ptr<uchar>(0);
		}
		
		inline float* getInputGrayPointerFloat()
		{
			return  imageGrayInputFloat_.get();
		}

		inline int getNumRows() 
		{ 
			return imageRGBAInput_.rows;
		}
		
		inline int getNumCols()
		{
			return imageRGBAInput_.cols;
		}

		inline size_t getNumPixels() 
		{
			return imageRGBAInput_.cols * imageRGBAInput_.rows;
		}

		void copyDeviceRGBAToHostRGBAOut(uchar4* devicePointer);

		void copyDeviceGrayToHostGrayOut(uchar* devicePointer);

		void saveRGBAImgOut(const string& filename);

		void saveGrayImgOut(const string& filename);

	private:
		//attributes
		cv::Mat imageRGBAInput_;
		cv::Mat imageGrayInput_;
		shared_ptr<float> imageGrayInputFloat_;

		cv::Mat imageRGBAOutput_;
		cv::Mat imageGrayOutput_;

	};




	template<typename T>
	__host__ __forceinline__ shared_ptr<T> makeArrayCudaHost(size_t size)
	{
		T* memory = nullptr;
		checkCudaErrors(cudaMallocHost((void **)&memory, size * sizeof(T)));
		return shared_ptr<T>(memory, [](T* p) {  checkCudaErrors(cudaFreeHost(p)); });
	}

	template<typename T>
	__host__ __forceinline__ shared_ptr<T> makeArray(size_t size)
	{
		return std::shared_ptr<T>( new T[size], [](T* p) { delete[] p; });
	}

	void deallocateMemmoryDevice(void* pointer);
	

	template <typename T>
	__host__ __forceinline__ shared_ptr<T> allocateMemmoryDevice(size_t size)
	{
		T* memory = nullptr;
		checkCudaErrors(cudaMalloc((void **)&memory, size * sizeof(T)));
		return shared_ptr<T>(memory, [](T* p) {  checkCudaErrors(cudaFree(p)); });
	}

	template <typename T>
	__host__ __forceinline__ T* allocateMemmoryDevicePointer(size_t size)
	{
		T* memory = nullptr;
		checkCudaErrors(cudaMalloc((void **)&memory, size * sizeof(T)));
		return memory;
	}


	template <typename T>
	__host__ __forceinline__ shared_ptr<T> allocateManagedMemory(size_t size)
	{
		T* memory = nullptr;
		checkCudaErrors(cudaMallocManaged((void **)&memory, size * sizeof(T)));
		return shared_ptr<T>(memory, [](T* p) {  checkCudaErrors(cudaFree(p)); });
	}

	template <typename T>
	__host__ __forceinline__ T* allocateManagedMemoryPointer(size_t size)
	{
		T* memory = nullptr;
		checkCudaErrors(cudaMallocManaged((void **)&memory, size * sizeof(T)));
		return memory;
	}

	template <typename T = void>
	__host__  __device__ void printFromKernel(const char *description, int what)
	{
		printf("%s: %d \n", description, what);
	}

	template <typename T = void>
	__host__  __device__ void printFromKernel(const char *description, double what)
	{
		printf("%s: %f \n", description, what);
	}

	template <typename T = void>
	__host__  __device__ void printFromKernel(const char *description, float what)
	{
		printf("%s: %f \n", description, what);
	}

	template <typename T = void>
	__host__ __device__ void printFromKernel(const char *description, float* what)
	{
		printf("%s: %p \n", description, what);
	}

	template <typename T = void>
	__host__  __device__ void printFromKernel(const char *description, uint what)
	{
		printf("%s: %d \n", description, what);
	}

	inline int roundUp(int numToRound, int multiple)
	{
		assert(multiple);
		return ((numToRound + multiple - 1) / multiple) * multiple;
	}

	shared_ptr<float> makeDeviceFilters(vector<shared_ptr<AbstractFilter>>& filters);
	
	shared_ptr<AbstractFilter> createFilter(uint width, vector<float> filter, const float multiplier = 1.0);

	shared_ptr<AbstractFilter> createFilter(uint width, float* filter, const float multiplier = 1.0);
	
	

}