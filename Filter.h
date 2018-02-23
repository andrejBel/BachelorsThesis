#ifndef FILTER_H
#define FILTER_H

#include <opencv2/core/core.hpp>
#include <vector>
#include <type_traits>


using namespace std;
namespace processing 
{
	template <typename T>
	class Filter
	{
		static_assert(std::is_floating_point<T>::value, "Class Filter can only be instantiazed with float, double or long double");

	public:
		Filter(uint width,uint height,  vector<T> filter,const T multiplier = 1.0);

		Filter(uint width, uint height,const T* filter,const T multiplier = 1.0);
		
		T* getHostFilterPointer();

		T* getDeviceFilterPointer();

		T getMultiplier();

		void allocateAndCopyHostFilterToDevice();

		void deallocateDeviceFilter();

		inline auto getFilterWidth() 
		{
			return width_;
		}

		inline auto getFilterHeight()
		{
			return height_;
		}

		~Filter();

	private:
		const T multiplier_;
		uint width_;
		uint height_;
		T* d_filter_;
		vector<T> h_filter_;


	};

}

#endif