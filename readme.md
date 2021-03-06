### Generating artificial data using parallel processing on GPU

The bachelor thesis is an implementation of 2D discrete convolution on GPU in CUDA. The text part of thesis contains some theory from image processing and the process of improving the implementation through several iterations. 


### Abstract

The aim of this bachelor thesis is generating artificial data using parallel processing on GPU. Based on analysis of methods used in image preprocessing, such method was selected, which is suitable for image processing and can be easily parallelized on GPU. When deciding, which method to implement, we took into consideration the availability and quality of existing CPU and GPU implementations of concrete method. After the analysis of selected methods of image preprocessing and the comparison of existing implementations, we decided to implement method called discrete two-dimensional convolution, which is fundamental for numerous tasks in image preprocessing and there are more kinds of convolution(1:1:1, 1:N:N, N:M:M), which will be discussed in detail in thesis. Our implementation had to satisfy the requirement to be implemented on GPU through CUDA programming model. Convolution is a computationally intensive operation and it is expected to significantly accelerate convolution by applying parallel processing on GPU compared to its CPU implementation. We made an effort to create GPU implementation more effective than existing one. GPU implementations are result of iterative process, where one can observe progress in every iteration. After experimental evaluation we found out that our GPU implementation is considerably faster, compared to CPU implementation from OpenCV and under certain circumstances 2-3 times faster than existing GPU OpenCL implementation from OpenCV, which is considered as standard in image processing.

### Code

Code is not in the state to be built,  the files are put in folders according to their purpose, watch, do not build.   
