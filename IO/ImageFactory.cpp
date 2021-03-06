#include "ImageFactory.h"
#include <iostream>
#include "MemoryPoolPinned.h"
#include "MemoryPoolManaged.h"

namespace processing 
{

	ImageFactory::ImageFactory(const string & fileName, MEMORY_TYPE type) :
		memoryType_(type)
	{
		imread(fileName, CV_LOAD_IMAGE_UNCHANGED).copyTo(imageGrayInput_);
		if (imageGrayInput_.empty()) {
			std::cerr << "Couldn't open image: " << fileName << std::endl;
			exit(1);
		}
		if (imageGrayInput_.channels() > 1)
		{
			cvtColor(imageGrayInput_, imageGrayInput_, CV_BGR2GRAY);
		}
		imageGrayInput_.convertTo(imageGrayInput_, CV_32FC1);
		switch (type)
		{
		case MEMORY_TYPE::MANAGED: 
		{
			const size_t numPixels = getNumPixels();
			imageGrayInputFloat_ = MemoryPoolManaged::getMemoryPoolManaged().acquireMemory(numPixels);
			std::copy((float *)imageGrayInput_.data, (float *)imageGrayInput_.data + numPixels, imageGrayInputFloat_.get());
			break;
		}
		case MEMORY_TYPE::PINNED:
		{
			const size_t numPixels = getNumPixels();
			imageGrayInputFloat_ = MemoryPoolPinned::getMemoryPoolPinnedForInput().acquireMemory(numPixels, true);
			std::copy((float *)imageGrayInput_.data, (float *)imageGrayInput_.data + numPixels, imageGrayInputFloat_.get());
			break;
		}
		case MEMORY_TYPE::NORMAL:
			break;
		default:
			throw std::runtime_error("Unknown memory type");
			break;
		}

	}

	void ImageFactory::saveImage(const string & filename, const ImageFactory & factory)
	{
		ImageFactory::saveImage(filename, factory.getNumCols(), factory.getNumRows(), factory.getInputGrayPointerFloat());
	}

	void ImageFactory::saveImage(const string & filename, int numCols, int numRows, float * data, const bool cropped, const int filterWidth)
	{			
		int toAdd(0);
		if ( cropped && filterWidth) 
		{
			toAdd = -filterWidth + 1;
		}
		Mat output_image(numRows + toAdd, numCols + toAdd, CV_32FC1, data);
		threshold(output_image, output_image, 0, 0, cv::THRESH_TOZERO);
		cv::normalize(output_image, output_image, 0.0, 255.0, cv::NORM_MINMAX);
		output_image.convertTo(output_image, CV_8UC1);
		cv::imwrite(filename, output_image);
	}

}



