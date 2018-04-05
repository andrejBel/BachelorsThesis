#include "ImageFactory.h"
#include <iostream>
#include "MemoryPoolPinned.h"

namespace processing 
{

	ImageFactory::ImageFactory(const string & fileName, bool memoryPool) :
		memoryPool_(memoryPool)
	{
		imread(fileName, CV_LOAD_IMAGE_UNCHANGED).copyTo(imageGrayInput_);
		if (imageGrayInput_.empty()) {
			std::cerr << "Couldn't open file: " << fileName << std::endl;
			exit(1);
		}
		if (imageGrayInput_.channels() > 1)
		{
			cvtColor(imageGrayInput_, imageGrayInput_, CV_BGR2GRAY);
		}
		imageGrayInput_.convertTo(imageGrayInput_, CV_32FC1);
		if (memoryPool)
		{
			imageGrayInputFloat_ = MemoryPoolPinned::getMemoryPoolPinnedForInput().acquireMemory();
			const size_t numPixels = getNumPixels();
			std::copy((float *)imageGrayInput_.data, (float *)imageGrayInput_.data + numPixels, imageGrayInputFloat_.get());
		}
	}

	void ImageFactory::saveImage(const string & filename, const ImageFactory & factory)
	{
		ImageFactory::saveImage(filename, factory.getNumCols(), factory.getNumRows(), factory.getInputGrayPointerFloat());
	}

	void ImageFactory::saveImage(const string & filename, int numCols, int numRows, float * data)
	{
		Mat output_image(numRows, numCols, CV_32FC1, data);
		float * fl = (float *)output_image.ptr<uchar>(0);
		cout << "Float" << endl;
		for (int i = 0; i < 10; i++)
		{
			cout << fl[0] << endl;
		}
		cout << "Uchar" << endl;
		threshold(output_image, output_image, 0, 0, cv::THRESH_TOZERO);
		cv::normalize(output_image, output_image, 0.0, 255.0, cv::NORM_MINMAX);

		output_image.convertTo(output_image, CV_8UC1);
		uchar * uch =  output_image.ptr<uchar>(0);
		for (int i = 0; i < 10; i++)
		{
			cout << (int) uch[0] << endl;
		}

		cv::imwrite(filename, output_image);
	}

}



