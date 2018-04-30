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

		Runnable(const bool multi, const bool cropped):
			multi_(multi),
			cropped_(cropped)
		{}

		DELETECOPYASSINGMENT(Runnable)

			virtual void run(vector<shared_ptr<ImageFactory>>& images, vector<shared_ptr<Filter>>& filters, vector<shared_ptr<float>>& results) = 0;

			virtual void run(vector<shared_ptr<ImageFactory>>& images, vector<vector<shared_ptr<Filter>>>& filters, vector<shared_ptr<float>>& results) = 0;
		

		// daj prec toto
		virtual void run(ImageFactory& image, vector<shared_ptr<Filter>>& filters, vector<shared_ptr<float>>& results) {};
		
		virtual string getDescription() = 0;

		virtual ~Runnable()	{}

		inline bool isMulti() const
		{
			return multi_;
		}

		inline bool isCropped() const 
		{
			return cropped_;
		}

	private:
		bool multi_;
		bool cropped_;

	};

	class SimpleRunnable: public Runnable
	{
	public:
		SimpleRunnable(const bool cropped) : Runnable(false, cropped) 
		{}

		DELETECOPYASSINGMENT(SimpleRunnable)

		virtual string getDescription() override 
		{
			if (isCropped()) 
			{
				return "Simple convolution cropped";
			}
			else 
			{
				return "Simple convolution";
			}
			
		}

		virtual void run(vector<shared_ptr<ImageFactory>>& images, vector<vector<shared_ptr<Filter>>>& filters, vector<shared_ptr<float>>& results) override 
		{
			throw runtime_error("Multi convolution not supported!");
		}

		~SimpleRunnable() 
		{}

	};

	class MultiRunnable : public Runnable
	{
	public:
		MultiRunnable() : Runnable(true, true)
		{}

		DELETECOPYASSINGMENT(MultiRunnable)

		virtual string getDescription() override
		{
			return "Multi convolution";
		}

		virtual void run(vector<shared_ptr<ImageFactory>>& images, vector<shared_ptr<Filter>>& filters, vector<shared_ptr<float>>& results) override
		{
			throw runtime_error("Single convolution not supported!");
		}

		~MultiRunnable()
		{}

	};




}



