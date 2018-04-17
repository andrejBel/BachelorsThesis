#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/core/utility.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <memory>

using namespace std;
using namespace cv;


namespace processing 
{
	class ImageFactory
	{
	public:
		ImageFactory(const string& fileName, const bool pinnedMemory = true);

		ImageFactory(const ImageFactory& other) = delete;

		ImageFactory& operator=(const ImageFactory& other) = delete;

		inline float* getInputGrayPointerFloat() const
		{
			return pinnedMemory_ ? imageGrayInputFloat_.get() : reinterpret_cast<float *>(imageGrayInput_.data);
		}

		inline int getNumRows() const
		{
			return imageGrayInput_.rows;
		}

		inline int getNumCols() const
		{
			return imageGrayInput_.cols;
		}

		inline size_t getNumPixels() const
		{
			return imageGrayInput_.cols * imageGrayInput_.rows;
		}

		static void saveImage(const string& filename, const ImageFactory& factory);

		static void saveImage(const string& filename, int numCols, int numRows, float* data, const bool cropped = false, const int filterWidth = 0);

	private:
		cv::Mat imageGrayInput_;
		const bool pinnedMemory_;
		shared_ptr<float> imageGrayInputFloat_;


	};
}
