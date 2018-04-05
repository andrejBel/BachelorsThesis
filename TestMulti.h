#pragma once

#include <string>
#include <vector>
#include <memory>
#include <iostream>
#include <algorithm>
#include "opencv2/core/utility.hpp"
#include "Runnable.h"
#include "Test.h"


namespace processing {

	class Filter;
	class TestMultiBuilder;
	class TestMulti
	{
		friend class TestMultiBuilder;
	public:
		TestMulti();
		
		vector<vector<shared_ptr<float>>> testCropped();
		static void testAlone(shared_ptr<Runnable> runnable, uint replications = 1);
		static void testAgainstEachOther(shared_ptr<Runnable> runnable1, shared_ptr<Runnable> runnable2, uint replications = 1);

	private:
		vector<vector<shared_ptr<Filter>>> filters_;
		vector<shared_ptr<Runnable>> runnables_;
		uint replications_;
		vector<string> fileNames_;
		float epsilon_;


	};

	class TestMultiBuilder
	{
	private:
		TestMulti test_;

	public:
		TestMultiBuilder& addFilterGroup(vector<shared_ptr<Filter>> filterGroup);

		TestMultiBuilder& setFilterGroups(vector<vector<shared_ptr<Filter>>> filters);

		TestMultiBuilder& addRunnable(shared_ptr<Runnable> runnable);

		TestMultiBuilder& setRunnables(vector<shared_ptr<Runnable>> runnables);

		TestMultiBuilder& setReplications(uint replications);

		TestMultiBuilder& addImagePath(const string& path);

		TestMultiBuilder& setImagePaths(const vector<string> paths);

		TestMultiBuilder& setEpsilon(float epsilon);

		TestMulti build();

	};



}



