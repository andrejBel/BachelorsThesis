#pragma once
#include <stdio.h>
#include <iostream>

#include "processing.h"

#include <vector>
#include <memory>

#include "KernelNaive.h"
#include "KernelSlowConvolution.h"
#include "KernelSlowConvolutionNoEdgeCopy.h"
#include "KernelSharedMemory.h"
#include "KernelSharedMemoryManaged.h"
#include "KernelSharedMemoryAsync.h"
#include "KernelSharedForSmall.h"
#include "KernelSharedThreads.h"

#include "Filter.h"
#include "CpuSlowConvolution.h"
#include "CPUSlowConvolutionAsync.h"
#include "CpuCropped.h"

#include <vector>
#include <algorithm>
#include "Test.h"
#include "ThreadPool.h"

#include "MemoryPoolPinned.h"

#include "MemoryPoolPitched.h"

using namespace cv;
using namespace std;
using namespace processing;


static const string OUTPUT_IMAGE_PATH = "output_img.jpg";

void sleep() 
{
	this_thread::sleep_for(chrono::milliseconds(2000));
}


int main()
{
	MemoryPoolPinned::getMemoryPoolPinnedForInput();
	MemoryPoolPinned::getMemoryPoolPinnedForOutput();
	MemoryPoolPitched::getMemoryPoolPitchedForInput();
	MemoryPoolPitched::getMemoryPoolPitchedForOutput();
	allocateMemmoryDevice<uchar>(1);
	
	

	//Test::testAgainstEachOther(make_shared<KernelSharedThreads>(), make_shared<CpuCropped>());
	Test::testAgainstEachOther(make_shared<KernelSharedThreads>(), make_shared<CpuCropped>());

	return 0;
}

