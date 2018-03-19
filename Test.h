#pragma once
#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <algorithm>
#include "opencv2/core/utility.hpp"
#include "Filter.h"
#include "Runnable.h"


using namespace std;
using namespace cv;

namespace processing 
{

	static const string INPUT_IMAGE_PATH = "input_img.jpg";

	class TestBuilder;
	class Test
	{
		friend class TestBuilder;
	public:

		Test();
		
		vector<vector<shared_ptr<float>>> operator()();
		vector<vector<shared_ptr<float>>> testForMannaged();
		vector<vector<shared_ptr<float>>> testCropped();
		

		static void testAlone(shared_ptr<Runnable> runnable, uint replications = 1);
		static void testAloneForManaged(shared_ptr<Runnable> runnable, uint replications = 1);
		static void testAgainstCpuMulticore(shared_ptr<Runnable> runnable, uint replications = 1);
		static void testAgainstCpuSingleCore(shared_ptr<Runnable> runnable, uint replications = 1);
		static void testAgainstEachOther(shared_ptr<Runnable> runnable1, shared_ptr<Runnable> runnable2, uint replications = 1, bool cropped = true);


		static void testAllAgainstCpu();


	private:
		vector<shared_ptr<AbstractFilter>> filters_;
		vector<shared_ptr<Runnable>> runnables_;
		uint replications_;
		string fileName_;
		float epsilon_;
	
	public:
		//excel generate command
		//=CONCAT(SUBSTITUTE(TEXT(ROUND(RAND()*RAND()*0,4;6);"0,000000");",";".";1);"f,")
		static shared_ptr<AbstractFilter> get1x1Filter();
		
		static shared_ptr<AbstractFilter> get3x3Filter();

		static shared_ptr<AbstractFilter> get5x5Filter();

		static shared_ptr<AbstractFilter> get7x7Filter();

		static shared_ptr<AbstractFilter> get9x9Filter();

		static shared_ptr<AbstractFilter> get11x11Filter();

		static shared_ptr<AbstractFilter> get13x13Filter();

		static shared_ptr<AbstractFilter> get15x15Filter();
		
		

	};


	class TestBuilder
	{
	private:
		Test test_;

	public:
		TestBuilder& addFilter(shared_ptr<AbstractFilter> filter);


		TestBuilder& setFilters(vector<shared_ptr<AbstractFilter>> filters);


		TestBuilder& addRunnable(shared_ptr<Runnable> runnable);


		TestBuilder& setRunnables(vector<shared_ptr<Runnable>> runnables);


		TestBuilder& setReplications(uint replications);


		TestBuilder& setImagePath(const string& path);


		TestBuilder& setEpsilon(float epsilon);

		Test build();

	};



}


