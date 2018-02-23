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



using namespace std;
using namespace cv;



namespace processing 
{
	class Runnable;
	class ImageFactory
	{
	public:
		ImageFactory(const string& fileName);

		ImageFactory(const ImageFactory& other) = delete;

		ImageFactory& operator=(const ImageFactory& other) = delete;

		~ImageFactory();

		inline uchar4* getHostRGBAPointer() 
		{
			return reinterpret_cast<uchar4 *>(h_imageRGBA_.ptr<uchar>(0));
		}

		inline uchar4* getDeviceRGBAPointer() 
		{
			return d_rGBAPointer_;
		}

		inline uchar* getHostGrayPointer()
		{
			return h_imageGray_.ptr<uchar>(0);
		}

		inline uchar* getDeviceGrayPointer()
		{
			return d_grayPointer_;
		}

		inline size_t getNumRows() 
		{ 
			return h_imageRGBA_.rows; 
		}
		
		inline size_t getNumCols() 
		{
			return h_imageRGBA_.cols; 
		}

		Mat& getHostRGBA() 
		{
			return h_imageRGBA_;
		}

		Mat& getHostGray()
		{
			return h_imageGray_;
		}

		void copyHostRGBAToDeviceRGBA();

		void copyDeviceRGBAToHostRGBA();

		void copyHostGrayToDeviceGray();

		void copyDeviceGrayToHostGray();

		void copyDeviceGrayToHostGray(uchar* devicePointer);

		void saveRGBAImg(const string& filename);

		void saveGrayImg(const string& filename);

		TickMeter run(Runnable* r);

	private:
		//attributes
		cv::Mat h_imageRGBA_;
		cv::Mat h_imageGray_;

		uchar4* d_rGBAPointer_;
		uchar* d_grayPointer_;

		//functions

	};


	void deallocateMemmoryDevice(void* pointer);
	

	template <typename T>
	T* allocateMemmoryDevice(size_t size)
	{
		T* memory = nullptr;
		checkCudaErrors(cudaMalloc((void **)&memory, size * sizeof(T)));
		return memory;
	}

	


}