#include "processing.h"
#include "Runnable.h"
#include <algorithm>
#include "MemoryPoolPinned.h"

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

		
		imageGrayInputFloat_ = MemoryPoolPinned::getMemoryPoolPinnedForInput().acquireMemory();
		const size_t numPixels = imageRGBAInput_.rows * imageRGBAInput_.cols;
		std::copy(imageGrayInput_.data, imageGrayInput_.data + numPixels, imageGrayInputFloat_.get());
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

	shared_ptr<float> makeDeviceFilters(vector<shared_ptr<AbstractFilter>>& filters)
	{
		size_t memmoryToAllocateForFiltersOnDevice(0);
		for_each(filters.begin(), filters.end(), [&memmoryToAllocateForFiltersOnDevice](auto& filter) { memmoryToAllocateForFiltersOnDevice += filter->getSize(); });
		shared_ptr<float> hostFilters = makeArray<float>(memmoryToAllocateForFiltersOnDevice);

		shared_ptr<float> deviceFilters = allocateMemmoryDevice<float>(memmoryToAllocateForFiltersOnDevice);
		uint offset(0);

		for_each(filters.begin(), filters.end(), [&hostFilters, &offset](auto& filter)
		{
			memcpy(hostFilters.get() + offset, filter->getFilter(), filter->getSize()*sizeof(float));
			offset += filter->getSize();
		});
		checkCudaErrors(cudaMemcpy(deviceFilters.get(), hostFilters.get(), memmoryToAllocateForFiltersOnDevice * sizeof(float), cudaMemcpyHostToDevice));
		return deviceFilters;
	}

	shared_ptr<AbstractFilter> createFilter(uint width, vector<float> filter, const float multiplier)
	{
		switch (width)
		{
		case 1: return make_shared<Filter<1>>(filter, multiplier);
		case 3: return make_shared<Filter<3>>(filter, multiplier);
		case 5: return make_shared<Filter<5>>(filter, multiplier);
		case 7: return make_shared<Filter<7>>(filter, multiplier);
		case 9: return make_shared<Filter<9>>(filter, multiplier);
		case 11: return make_shared<Filter<11>>(filter, multiplier);
		case 13: return make_shared<Filter<13>>(filter, multiplier);
		case 15: return make_shared<Filter<15>>(filter, multiplier);
		default:
			std::cerr << "Filter with width:" << width << "not supported!" << endl;
			break;
		}
		return shared_ptr<AbstractFilter>();
	}

	shared_ptr<AbstractFilter> createFilter(uint width, float * filter, const float multiplier)
	{
		vector<float> filterVec(filter, filter + width);
		return createFilter(width, filterVec, multiplier);
	}


}
