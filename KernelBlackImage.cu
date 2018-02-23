#include "KernelBlackImage.h"

#include "device_launch_parameters.h"

#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "processing.h"



using namespace cv;
using namespace std;




__global__ void emptyKernel()
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	printf("Grid X: %d, Grid Y: %d, Grid Z: %d\n", gridDim.x, gridDim.y, gridDim.z);
	printf("Block X: %d, Block Y: %d, Block Z: %d\n", blockIdx.x, blockIdx.y, blockIdx.z);
	printf("Thread X: %d, Thread Y: %d, Thread Z: %d\n", threadIdx.x, threadIdx.y, threadIdx.z);
	printf("Block dim X: %d, Block dim Y: %d, Block dim Z: %d\n", blockDim.x, blockDim.y, blockDim.z);
}


namespace processing
{

	__global__ void nullGray(uchar * grayPtr, const size_t numRows, const size_t numCols)
	{
		int absolute_image_position_x = blockIdx.x * blockDim.x + threadIdx.x;
		int absolute_image_position_y = blockIdx.y * blockDim.y + threadIdx.y;
		if (absolute_image_position_x >= numCols || absolute_image_position_y >= numRows)
		{
			return;
		}
		const int index_1D = absolute_image_position_y * numCols + absolute_image_position_x;
		grayPtr[index_1D] = 256- 1 - grayPtr[index_1D];
	}

	KernelBlackImage::KernelBlackImage()
	{
	}

	KernelBlackImage::~KernelBlackImage()
	{
	}

	void KernelBlackImage::run(ImageFactory & image)
	{

		const uint numberOfThreadsInBlock = 32;
		const dim3 blockSize(numberOfThreadsInBlock, numberOfThreadsInBlock, 1);
		const dim3 gridSize((image.getNumCols() + blockSize.x - 1) / blockSize.x, (image.getNumRows() + blockSize.y - 1) / blockSize.y, 1);
		
		nullGray <<<gridSize, blockSize >>> (image.getDeviceGrayPointer(), image.getNumRows(), image.getNumCols());
	}



}

