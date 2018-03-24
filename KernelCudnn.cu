#include "KernelCudnn.h"

#include <cuda.h>
#include <helper_cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include "opencv2/core/utility.hpp"

#include <vector>
#include <cstdio>
#include <cmath>
#include <iostream>

#include <thread>
#include <algorithm>

#include <cudnn.h>

#include "opencv2/core.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

#include <opencv2\cudaimgproc.hpp>

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

namespace processing {

	KernelCudnn::KernelCudnn()
	{
	}



	void KernelCudnn::run(ImageFactory & image, vector<shared_ptr<AbstractFilter>>& filters, vector<shared_ptr<float>>& results)
	{
		shared_ptr<float> deviceFilters = makeDeviceFilters(filters);

		// filter allocation and initialization
		shared_ptr<float> deviceGrayImageOut = allocateMemmoryDevice<float>(image.getNumPixels());
		const float * hostGrayImage = image.getInputGrayPointerFloat();

		shared_ptr<float> deviceGrayImageIn = allocateMemmoryDevice<float>(image.getNumPixels());
		checkCudaErrors(cudaMemcpy(deviceGrayImageIn.get(), hostGrayImage, image.getNumPixels() * sizeof(float), cudaMemcpyHostToDevice));
		// memory allocation

		const uint numberOfThreadsInBlock = 16;
		const dim3 blockSize(numberOfThreadsInBlock, numberOfThreadsInBlock);
		const dim3 gridSize((image.getNumCols() + blockSize.x - 1) / blockSize.x, (image.getNumRows() + blockSize.y - 1) / blockSize.y, 1);


		cudnnHandle_t cudnn;
		cudnnCreate(&cudnn);
		cudnnTensorDescriptor_t input_descriptor;
		checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
		checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
			/*format=*/CUDNN_TENSOR_NHWC,
			/*dataType=*/CUDNN_DATA_FLOAT,
			/*batch_size=*/1,
			/*channels=*/1,
			/*image_height=*/image.getNumRows(),
			/*image_width=*/image.getNumCols()));

		cudnnTensorDescriptor_t output_descriptor;
		checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
		checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
				/*format=*/CUDNN_TENSOR_NHWC,
				/*dataType=*/CUDNN_DATA_FLOAT,
				/*batch_size=*/1,
				/*channels=*/1,
				/*image_height=*/image.getNumRows(),
				/*image_width=*/image.getNumCols()));

		cudnnFilterDescriptor_t kernel_descriptor;
		checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
		checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
				/*dataType=*/CUDNN_DATA_FLOAT,
				/*format=*/CUDNN_TENSOR_NCHW,
				/*out_channels=*/1,
				/*in_channels=*/1,
				/*kernel_height=*/3,
				/*kernel_width=*/3));
		cudnnConvolutionDescriptor_t convolution_descriptor;
		checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
		checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
			/*pad_height=*/1,
			/*pad_width=*/1,
			/*vertical_stride=*/1,
			/*horizontal_stride=*/1,
			/*dilation_height=*/1,
			/*dilation_width=*/1,
			/*mode=*/cudnnConvolutionMode_t::CUDNN_CONVOLUTION,
			/*computeType=*/CUDNN_DATA_FLOAT));

		cudnnConvolutionFwdAlgo_t convolution_algorithm;
		checkCUDNN(
			cudnnGetConvolutionForwardAlgorithm(cudnn,
				input_descriptor,
				kernel_descriptor,
				convolution_descriptor,
				output_descriptor,
				CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
				/*memoryLimitInBytes=*/0,
				&convolution_algorithm))

		size_t workspace_bytes = 0;
		checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
			input_descriptor,
			kernel_descriptor,
			convolution_descriptor,
			output_descriptor,
			convolution_algorithm,
			&workspace_bytes));
		std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB"
			<< std::endl;
		void* d_workspace{ nullptr };
		cudaMalloc(&d_workspace, workspace_bytes);

		// Mystery kernel
		const float kernel_template[3][3] = {
			{ 1.0f,  2.0f, 3.3f },
			{ 4.6f, -8.0f, 5.7f },
			{ 6.5f,  7.1f, 8.9f }
		};

		float h_kernel[1][1][3][3];
		for (int kernel = 0; kernel < 1; ++kernel) {
			for (int channel = 0; channel < 1; ++channel) {
				for (int row = 0; row < 3; ++row) {
					for (int column = 0; column < 3; ++column) {
						h_kernel[kernel][channel][row][column] = kernel_template[row][column];
					}
				}
			}
		}

		float* d_kernel{ nullptr };
		cudaMalloc(&d_kernel, sizeof(h_kernel));
		cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);



		const float alpha = 1, beta = 0;
		checkCUDNN(cudnnConvolutionForward(cudnn,
			&alpha,
			input_descriptor,
			deviceGrayImageIn.get(),
			kernel_descriptor,
			d_kernel,
			convolution_descriptor,
			convolution_algorithm,
			d_workspace,
			workspace_bytes,
			&beta,
			output_descriptor,
			deviceGrayImageOut.get()));


		cudaFree(d_kernel);
		cudaFree(d_workspace);

		cudnnDestroyTensorDescriptor(input_descriptor);
		cudnnDestroyTensorDescriptor(output_descriptor);
		cudnnDestroyFilterDescriptor(kernel_descriptor);
		cudnnDestroyConvolutionDescriptor(convolution_descriptor);
		cudnnDestroy(cudnn);


		Ptr<cuda::Convolution> convolver = cuda::createConvolution(Size(32, 16));

		float filter[] = { 1,2,3,4,5,6,7,8,9 };
		cv::Mat kernel(3 ,3, CV_32FC1,(void *) filter);
		cv::Mat input(image.getNumRows(), image.getNumCols(), CV_32FC1, image.getInputGrayPointerFloat());
		cv::Mat output(image.getNumRows(), image.getNumCols(), CV_32FC1);
		convolver->convolve(input, kernel, output);
		


		uint offset(0);
		for (auto& filter : filters)
		{
			switch (filter->getWidth())
			{
			case 3:
			{
				float * ptr = (deviceFilters.get() + offset);
				


				//convolutionGPUNaive << <gridSize, blockSize >> >(ptr, image.getNumRows(), image.getNumCols(), deviceGrayImageIn.get(), deviceGrayImageOut.get(), 1);
				break;
			}
			
			default:
				std::cerr << "Filter with width: " << filter->getWidth() << " not supported!" << endl;
				break;
			}
			offset += filter->getSize();
			shared_ptr<float> resultCPU = makeArray<float>(image.getNumPixels());
			checkCudaErrors(cudaMemcpy(resultCPU.get(), deviceGrayImageOut.get(), image.getNumPixels() * sizeof(float), cudaMemcpyDeviceToHost));
			results.push_back(resultCPU);
		}
		checkCudaErrors(cudaDeviceSynchronize());
	
	}

}



