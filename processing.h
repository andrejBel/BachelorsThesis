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

using namespace std;
using namespace cv;



namespace processing 
{
	template <typename T>
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

		template <typename T>
		TickMeter run(Runnable<T>* r, vector<shared_ptr<T>>& results) 
		{
			TickMeter timer;
			timer.start();
			r->run(*this, results);
			timer.stop();
			return timer;
		}


	private:
		//attributes
		cv::Mat imageRGBAInput_;
		cv::Mat imageGrayInput_;

		cv::Mat imageRGBAOutput_;
		cv::Mat imageGrayOutput_;

	};


	template<typename T>
	inline shared_ptr<T> makeArray(size_t size)
	{
		return std::shared_ptr<T>(new T[size], [](T* p) { delete[] p; });
	}

	void deallocateMemmoryDevice(void* pointer);
	

	template <typename T>
	shared_ptr<T> allocateMemmoryDevice(size_t size)
	{
		T* memory = nullptr;
		checkCudaErrors(cudaMalloc((void **)&memory, size * sizeof(T)));
		return shared_ptr<T>(memory, [](T* p) {  checkCudaErrors(cudaFree(p)); });
	}

	


}