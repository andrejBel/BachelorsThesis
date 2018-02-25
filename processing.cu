#include "processing.h"
#include "Runnable.h"

namespace processing 
{

	ImageFactory::ImageFactory(const string & fileName)
	{
		Mat image = cv::imread(fileName, CV_LOAD_IMAGE_COLOR);
		if (image.empty()) {
			std::cerr << "Couldn't open file: " << fileName << std::endl;
			exit(1);
		}
		cv::cvtColor(image, imageRGBAInput_, CV_BGR2RGBA);
		cv::cvtColor(imageRGBAInput_, imageGrayInput_, CV_RGBA2GRAY);

		const size_t numPixels = imageRGBAInput_.rows * imageRGBAInput_.cols;
		imageRGBAOutput_.create(imageRGBAInput_.rows, imageRGBAInput_.cols, CV_8UC4);
		imageGrayOutput_.create(imageRGBAInput_.rows, imageRGBAInput_.cols, CV_8UC1);
	}

	void ImageFactory::copyDeviceRGBAToHostRGBAOut(uchar4 * devicePointer)
	{
		checkCudaErrors(cudaMemcpy(getOutputRGBAPointer(), devicePointer, imageRGBAOutput_.rows * imageRGBAOutput_.cols * sizeof(uchar4), cudaMemcpyDeviceToHost));
	}

	void ImageFactory::copyDeviceGrayToHostGrayOut(uchar * devicePointer)
	{
		checkCudaErrors(cudaMemcpy(getOutputGrayPointer(), devicePointer, imageGrayOutput_.rows * imageGrayOutput_.cols * sizeof(uchar), cudaMemcpyDeviceToHost));
	}

	void ImageFactory::saveRGBAImgOut(const string & filename)
	{
		Mat outPut(imageRGBAOutput_.rows, imageRGBAOutput_.cols, CV_8UC4, getOutputRGBAPointer());
		cv::cvtColor(outPut, outPut, CV_RGBA2BGR);
		cv::imwrite(filename.c_str(), outPut);
	}

	void ImageFactory::saveGrayImgOut(const string & filename)
	{
		cv::imwrite(filename.c_str(), imageGrayOutput_);
	}

	void deallocateMemmoryDevice(void * pointer)
	{
		checkCudaErrors(cudaFree(pointer));
	}

}
