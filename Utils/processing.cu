#include "processing.h"
#include "Runnable.h"
#include <algorithm>
#include "ImageFactory.h"
#include "Filter.h"
namespace processing 
{

	

	void deallocateMemmoryDevice(void * pointer)
	{
		checkCudaErrors(cudaFree(pointer));
	}

	shared_ptr<float> makeDeviceFilters(vector<shared_ptr<Filter>>& filters)
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

	shared_ptr<Filter> createFilter(const uint width, const vector<float>& filter, const float multiplier)
	{
		return make_shared<Filter>(width, filter, multiplier);
	}

	shared_ptr<Filter> createFilter(const uint width, const float * filter, const float multiplier)
	{
		vector<float> filterVec(filter, filter + width * width);
		return createFilter(width, filterVec, multiplier);
	}

	pair<bool, string> controlInputForMultiConvolution(vector<shared_ptr<ImageFactory>>& images, vector<vector<shared_ptr<Filter>>>& filters)
	{
		size_t imageSize = images.size();
		if (imageSize == 0)
		{
			return make_pair(false, "No images to proces");
		}
		int numCols = images[0]->getNumCols(); //x
		int numRows = images[0]->getNumRows(); //y
		size_t pixels = numCols * numRows;
		for (size_t i = 1; i < imageSize; ++i)
		{
			if (images[i]->getNumCols() != numCols || images[i]->getNumRows() != numRows)
			{
				return  make_pair(false, "Images must have the same dimension");
			}
		}
		for (size_t i = 0; i < filters.size(); ++i)
		{
			if (filters[i].size() != imageSize)
			{
				return make_pair(false, "Filter group size is different than image size");
			}
			int filterWidth = filters[i][0]->getWidth();
			for (size_t j = 1; j < imageSize; ++j)
			{
				if (filters[i][j]->getWidth() != filterWidth)
				{
					return make_pair(false, "Filter in group are of different size");
				}
			}
		}
		return make_pair(true, "OK");
	}


}
