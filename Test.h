#pragma once
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <algorithm>
#include "opencv2/core/utility.hpp"
#include "Runnable.h"


using namespace std;
using namespace cv;

namespace processing 
{

	static const string INPUT_IMAGE_PATHS[] = { "input_img.jpg", "input_img small.jpg", "input_img skrabance.jpg", "input_img.jpg","input_img.jpg","input_img.jpg","input_img.jpg","input_img small.jpg","input_img small.jpg" };

	class Filter;
	class TestBuilder;
	class Test
	{
		friend class TestBuilder;
	public:

		Test();
		
		vector<vector<shared_ptr<float>>> testExtend();
		vector<vector<shared_ptr<float>>> testCropped();
		
		void run();

		static void testAlone(shared_ptr<Runnable> runnable, uint replications = 1);
		static void testAgainstEachOther(shared_ptr<Runnable> runnable1, shared_ptr<Runnable> runnable2, uint replications = 1);
		static void testAgainstCpu(shared_ptr<Runnable> runnable, uint replications = 1);

		static void testAllAgainstCpu();


	private:
		vector<shared_ptr<Filter>> filters_;
		vector<shared_ptr<Runnable>> runnables_;
		uint replications_;
		vector<string> fileNames_;
		float epsilon_;
	
	public:
		//excel generate command
		//=CONCAT(SUBSTITUTE(TEXT(ROUND(RAND()*RAND()*0,4;6);"0,000000");",";".";1);"f,")
		static shared_ptr<Filter> get1x1Filter();
		
		static shared_ptr<Filter> get3x3Filter();
		static shared_ptr<Filter> Test::get3x3Filter1();
		static shared_ptr<Filter> Test::get3x3Filter2();
		static shared_ptr<Filter> Test::get3x3Filter3();
		static shared_ptr<Filter> Test::get3x3Filter4();
		static shared_ptr<Filter> Test::get3x3Filter5();

		static shared_ptr<Filter> get5x5Filter();

		static shared_ptr<Filter> get7x7Filter();

		static shared_ptr<Filter> get9x9Filter();

		static shared_ptr<Filter> get11x11Filter();

		static shared_ptr<Filter> get13x13Filter();

		static shared_ptr<Filter> get15x15Filter();
		
		static shared_ptr<Filter> get17x17Filter();
		

	};


	class TestBuilder
	{
	private:
		Test test_;

	public:
		TestBuilder& addFilter(shared_ptr<Filter> filter);


		TestBuilder& setFilters(vector<shared_ptr<Filter>> filters);


		TestBuilder& addRunnable(shared_ptr<Runnable> runnable);


		TestBuilder& setRunnables(vector<shared_ptr<Runnable>> runnables);


		TestBuilder& setReplications(uint replications);


		TestBuilder& addImagePath(string imagePath);

		TestBuilder& setImagePaths(vector<string> imagePaths);


		TestBuilder& setEpsilon(float epsilon);

		Test build();

	};



}


