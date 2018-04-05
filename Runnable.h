#pragma once
#include <memory>
#include <vector>
#include <string>
#include "Filter.h"
#include "ImageFactory.h"

#define DELETECOPYASSINGMENT(CLASS)  \
	CLASS(const CLASS& other) = delete; \
    CLASS& operator=(const CLASS& other) = delete;

using namespace std;
namespace processing 
{
	class ImageFactory;

	class Runnable
	{
	public:

		Runnable()
		{}

		DELETECOPYASSINGMENT(Runnable)

		virtual void run(ImageFactory& image, vector<shared_ptr<Filter>>& filters, vector<shared_ptr<float>>& results) = 0;

		virtual void run(vector<shared_ptr<ImageFactory>>& images, vector<vector<shared_ptr<Filter>>>& filters, vector<shared_ptr<float>>& results)
		{
					
		}

		virtual string getDescription() = 0;

		virtual ~Runnable()	{}



	};
}



