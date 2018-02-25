#pragma once
#include <memory>
#include <vector>

#define DELETECOPYASSINGMENT(CLASS)  \
	CLASS(const CLASS& other) = delete; \
    CLASS& operator=(const CLASS& other) = delete;

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



