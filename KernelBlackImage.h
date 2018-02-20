#pragma once
#include "Runnable.h"

namespace processing
{
	class KernelBlackImage : public Runnable
	{
	public:
		KernelBlackImage();

		DELETECOPYASSINGMENT(KernelBlackImage)

			virtual ~KernelBlackImage();


		virtual void run(ImageFactory & image) override;

	};

}



