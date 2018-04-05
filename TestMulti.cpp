#include "TestMulti.h"
#include "processing.h"
#include "Filter.h"

namespace processing 
{
	TestMulti::TestMulti(): 
		epsilon_(0.001f),
		replications_(1)
	{}

	vector<vector<shared_ptr<float>>> processing::TestMulti::testCropped()
	{
		vector<vector<shared_ptr<float>>> results;
		vector<shared_ptr<ImageFactory>> images(this->fileNames_.size());

		for (size_t i = 0; i < images.size(); i++)
		{
			images[i] = make_shared<ImageFactory>(this->fileNames_[i]);
		}
		pair<bool, string> check = controlInputForMultiConvolution(images, this->filters_);
		if (check.first == false) 
		{
			cerr << check.second << endl;
			return results;
		}
		TickMeter meter;
		for (shared_ptr<Runnable> & runnable : runnables_)
		{
			double time(0);
			results.push_back(vector<shared_ptr<float>>());
			for (uint i = 0; i < replications_; i++)
			{
				meter.reset();
				meter.start();
				runnable->run(images, filters_, results[results.size() - 1]);
				meter.stop();
				time += meter.getTimeMilli();
			}
			cout << runnable->getDescription() << ". Replications:  " << replications_ << ". Average time: " << (time / replications_) << endl;
		}
		int columns = images[0]->getNumCols();
		int rows = images[0]->getNumRows();
		if (runnables_.size() == 2)
		{
			vector<shared_ptr<float>>& resultsFirst = results[0];
			vector<shared_ptr<float>>& resultsSecond = results[1];
			size_t pixels = images[0]->getNumPixels();
			auto size = std::min(resultsFirst.size(), resultsSecond.size());
			for (size_t i = 0; i < size; i++)
			{
				shared_ptr<float> resultFirst = resultsFirst[i];
				shared_ptr<float> resultSecond = resultsSecond[i];
				cout << "Bezim" << endl;
				size_t range = (columns - (filters_[i][0]->getWidth() - 1)) * (rows - (filters_[i][0]->getWidth() - 1));
				for (size_t j = 0; j < range; j++)
				{
					if ([&resultFirst, &resultSecond, j]() {
						return fabs(resultFirst.get()[j] - resultSecond.get()[j]) > 0.1;
					}())
					{

						cout << "-----------------------" << endl;
						cout << "Index: " << j << ", epsilon: " << epsilon_ << endl;
						cout << runnables_[0]->getDescription() << ": " << resultFirst.get()[j] << endl;
						cout << runnables_[1]->getDescription() << ": " << resultSecond.get()[j] << endl;
						cout << runnables_[0]->getDescription() << " - " << runnables_[1]->getDescription() << ": " << resultFirst.get()[j] - resultSecond.get()[j] << endl;
						cout << "-----------------------" << endl;

					}
				}
			}
		}
		return results;
	}

	void TestMulti::testAlone(shared_ptr<Runnable> runnable, uint replications)
	{
		TestMultiBuilder builder;
		builder
			
			.addFilterGroup(vector<shared_ptr<Filter>>({ Test::get3x3Filter(),Test::get3x3Filter(),Test::get3x3Filter() }))
			.addFilterGroup(vector<shared_ptr<Filter>>({ Test::get15x15Filter(),Test::get15x15Filter(),Test::get15x15Filter() }))
			.setImagePaths(vector<string>({ INPUT_IMAGE_PATH , INPUT_IMAGE_PATH , INPUT_IMAGE_PATH }))
			.addRunnable(runnable)
			.setReplications(replications);
		builder.build().testCropped();
	}

	void TestMulti::testAgainstEachOther(shared_ptr<Runnable> runnable1, shared_ptr<Runnable> runnable2, uint replications)
	{
		TestMultiBuilder builder;
		builder
			.addFilterGroup(vector<shared_ptr<Filter>>({ Test::get1x1Filter(),Test::get1x1Filter(),Test::get1x1Filter() }))
			.addFilterGroup(vector<shared_ptr<Filter>>({ Test::get3x3Filter(),Test::get3x3Filter(),Test::get3x3Filter() }))
			.addFilterGroup(vector<shared_ptr<Filter>>({ Test::get5x5Filter(),Test::get5x5Filter(),Test::get5x5Filter() }))
			.addFilterGroup(vector<shared_ptr<Filter>>({ Test::get7x7Filter(),Test::get7x7Filter(),Test::get7x7Filter() }))
			.addFilterGroup(vector<shared_ptr<Filter>>({ Test::get9x9Filter(),Test::get9x9Filter(),Test::get9x9Filter() }))
			.addFilterGroup(vector<shared_ptr<Filter>>({ Test::get11x11Filter(),Test::get11x11Filter(),Test::get11x11Filter() }))
			.addFilterGroup(vector<shared_ptr<Filter>>({ Test::get13x13Filter(),Test::get13x13Filter(),Test::get13x13Filter() }))
			.addFilterGroup(vector<shared_ptr<Filter>>({ Test::get15x15Filter(),Test::get15x15Filter(),Test::get15x15Filter() }))
			.setImagePaths(vector<string>({ INPUT_IMAGE_PATH , INPUT_IMAGE_PATH , INPUT_IMAGE_PATH }))
			.addRunnable(runnable1)
			.addRunnable(runnable2)
			.setReplications(replications);
		builder.build().testCropped();
	}

	TestMultiBuilder & TestMultiBuilder::addFilterGroup(vector<shared_ptr<Filter>> filterGroup)
	{
		test_.filters_.push_back(filterGroup);
		return *this;
	}

	TestMultiBuilder & TestMultiBuilder::setFilterGroups(vector<vector<shared_ptr<Filter>>> filters)
	{
		test_.filters_ = filters;
		return *this;
	}

	TestMultiBuilder & TestMultiBuilder::addRunnable(shared_ptr<Runnable> runnable)
	{
		test_.runnables_.push_back(runnable);
		return *this;
	}

	TestMultiBuilder & TestMultiBuilder::setRunnables(vector<shared_ptr<Runnable>> runnables)
	{
		test_.runnables_ = runnables;
		return *this;
	}

	TestMultiBuilder & TestMultiBuilder::setReplications(uint replications)
	{
		test_.replications_ = replications;
		return *this;
	}

	TestMultiBuilder & TestMultiBuilder::addImagePath(const string & path)
	{
		test_.fileNames_.push_back(path);
		return *this;
	}

	TestMultiBuilder & TestMultiBuilder::setImagePaths(const vector<string> paths)
	{
		test_.fileNames_ = paths;
		return *this;
	}

	TestMultiBuilder & TestMultiBuilder::setEpsilon(float epsilon)
	{
		test_.epsilon_ = epsilon;
		return *this;
	}

	TestMulti TestMultiBuilder::build()
	{
		return test_;
	}



}


