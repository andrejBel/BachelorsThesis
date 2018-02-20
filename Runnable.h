#pragma once

#define DELETECOPYASSINGMENT(CLASS)  \
	CLASS(const CLASS& other) = delete; \
    CLASS& operator=(const CLASS& other) = delete;
namespace processing 
{
	class ImageFactory;

	class Runnable
	{
	public:

		Runnable()
		{}

		DELETECOPYASSINGMENT(Runnable)

		virtual void run(ImageFactory& image) = 0;

		virtual ~Runnable()
		{
		}
	};
}



