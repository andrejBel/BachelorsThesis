#include "processing.h"
#include "Runnable.h"

namespace processing 
{

	ImageFactory::ImageFactory(const string & fileName):
		d_rGBAPointer_(nullptr),
		d_grayPointer_(nullptr)
	{
		h_imageRGBA_ = cv::imread(fileName, CV_LOAD_IMAGE_COLOR);
		if (h_imageRGBA_.empty()) {
			std::cerr << "Couldn't open file: " << fileName << std::endl;
			exit(1);
		}

		cv::cvtColor(h_imageRGBA_, h_imageRGBA_, CV_BGR2RGBA);
		cv::cvtColor(h_imageRGBA_, h_imageGray_, CV_RGBA2GRAY);

		const size_t numPixels = h_imageRGBA_.rows * h_imageRGBA_.cols;

		d_rGBAPointer_ = allocateMemmoryDevice<uchar4>(numPixels);
		d_grayPointer_ = allocateMemmoryDevice<uchar>(numPixels);

		copyHostRGBAToDeviceRGBA();
		copyHostGrayToDeviceGray();
	}

	ImageFactory::~ImageFactory()
	{
		deallocateMemmoryDevice(d_rGBAPointer_);
		deallocateMemmoryDevice(d_grayPointer_);
	}

	void ImageFactory::copyHostRGBAToDeviceRGBA()
	{
		checkCudaErrors(cudaMemcpy(d_rGBAPointer_, getHostRGBAPointer(), h_imageRGBA_.rows * h_imageRGBA_.cols * sizeof(uchar4), cudaMemcpyHostToDevice));
	}

	void ImageFactory::copyDeviceRGBAToHostRGBA()
	{
		checkCudaErrors(cudaMemcpy(getHostRGBAPointer(), d_rGBAPointer_, h_imageRGBA_.rows * h_imageRGBA_.cols * sizeof(uchar4), cudaMemcpyDeviceToHost));
	}

	void ImageFactory::copyHostGrayToDeviceGray()
	{
		checkCudaErrors(cudaMemcpy(d_grayPointer_, getHostGrayPointer(), h_imageGray_.rows * h_imageGray_.cols * sizeof(uchar), cudaMemcpyHostToDevice));
	}

	void ImageFactory::copyDeviceGrayToHostGray()
	{
		copyDeviceGrayToHostGray(d_grayPointer_);
		
	}

	void ImageFactory::copyDeviceGrayToHostGray(uchar * devicePointer)
	{
		checkCudaErrors(cudaMemcpy(getHostGrayPointer(), devicePointer, h_imageGray_.rows * h_imageGray_.cols * sizeof(uchar), cudaMemcpyDeviceToHost));
	}

	void ImageFactory::saveRGBAImg(const string & filename)
	{
		imwrite(filename, h_imageRGBA_);
	}

	void ImageFactory::saveGrayImg(const string & filename)
	{
		imwrite(filename, h_imageGray_);
	}

	TickMeter ImageFactory::run(Runnable * r)
	{
		TickMeter timer;
		timer.start();
		r->run(*this);
		timer.stop();
		return timer;
	}

	void deallocateMemmoryDevice(void * pointer)
	{
		checkCudaErrors(cudaFree(pointer));
	}

}
