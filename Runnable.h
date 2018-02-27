#pragma once
#include <memory>
#include <vector>

#define DELETECOPYASSINGMENT(CLASS)  \
	CLASS(const CLASS& other) = delete; \
    CLASS& operator=(const CLASS& other) = delete;

#define MERAJ (BLOCK,TILE) \
				{ \
					cv::TickMeter m;\
					Filter<T, 5> * ptr = (Filter<T, 5> *) (deviceFilters.get() + offset);\
					const int BLOCK_SIZE = BLOCK;\
					const int FILTER_WIDTH = 5;\
					const int TILE_SIZE = TILE;\
					const dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);\
					const dim3 gridSize((image.getNumCols() + TILE_SIZE - 1) / TILE_SIZE, (image.getNumRows() + TILE_SIZE - 1) / TILE_SIZE, 1);\
					m.start();\
					convolutionGPUShared<T, FILTER_WIDTH, BLOCK_SIZE, TILE_SIZE> << <gridSize, blockSize >> >(ptr, image.getNumRows(), image.getNumCols(), deviceGrayImageIn.get(), deviceGrayImageOut.get());\
					m.stop();\
					checkCudaErrors(cudaDeviceSynchronize()); \
					cout << "Block size: " << BLOCK_SIZE << ", TILE_SIZE: " << TILE_SIZE << endl << ", time: " << m.getTimeMicro(); \
				}



using namespace std;
namespace processing 
{
	class ImageFactory;

	template<typename T>
	class Runnable
	{
	public:

		Runnable()
		{}

		DELETECOPYASSINGMENT(Runnable<T>)

		virtual void run(ImageFactory& image, vector<shared_ptr<T>>& results) = 0;

		virtual ~Runnable()
		{
		}
	};
}



