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

using namespace std;
using namespace cv;



namespace processing 
{
	static const uint MAX_IMAGE_WIDTH = 4200;
	static const uint MAX_IMAGE_HEIGHT = 4200;
	static const size_t MAX_IMAGE_RESOLUTION = MAX_IMAGE_WIDTH*MAX_IMAGE_HEIGHT;
	static const uint PINNED_MEMORY_BUFFER_SIZE_INPUT = 5;
	static const uint PINNED_MEMORY_BUFFER_SIZE_OUTPUT = 20;
	static const uint PITCHED_MEMORY_BUFFER_SIZE_INPUT = 2;
	static const uint PITCHED_MEMORY_BUFFER_SIZE_OUTPUT = 10;


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

	const int MAXFILTERWIDTH = 15;
	static __constant__ float FILTERCUDA[MAXFILTERWIDTH * MAXFILTERWIDTH];


	template <typename T = void>
	__device__ void printFromKernel(const char *description, int what)
	{
		printf("%s: %d \n", description, what);
	}

	template <typename T = void>
	__device__ void printFromKernel(const char *description, double what)
	{
		printf("%s: %f \n", description, what);
	}

	template <typename T = void>
	__device__ void printFromKernel(const char *description, float what)
	{
		printf("%s: %f \n", description, what);
	}

	template <typename T = void>
	__device__ void printFromKernel(const char *description, uint what)
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