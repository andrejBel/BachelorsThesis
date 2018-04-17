#include "TestMulti.h"
#include "processing.h"
#include "Filter.h"
#include "MemoryPoolPinned.h"
namespace processing 
{
	TestMulti::TestMulti(): 
		epsilon_(2.0f),
		replications_(1)
	{}

	vector<vector<shared_ptr<float>>> processing::TestMulti::testCropped()
	{
		
		vector<vector<shared_ptr<float>>> results;
		vector<shared_ptr<ImageFactory>> images(this->fileNames_.size());
		shared_ptr<ImageFactory> im = make_shared<ImageFactory>(this->fileNames_[0]);
		for (size_t i = 0; i < images.size(); i++)
		{
			images[i] = im;
		}
		pair<bool, string> check = controlInputForMultiConvolution(images, this->filters_);
		if (check.first == false) 
		{
			cerr << check.second << endl;
			return results;
		}
		std::sort(this->filters_.begin(), this->filters_.end(), [](vector<shared_ptr<Filter>>& first, vector<shared_ptr<Filter>>& second)
		{
			return first[0]->getWidth() < second[0]->getWidth();
		});
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
					if ([&resultFirst, &resultSecond, j, this]() {
						return fabs(resultFirst.get()[j] - resultSecond.get()[j]) > epsilon_;
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

	void TestMulti::run()
	{
		if (runnables_.size() == 1)
		{
			shared_ptr<Runnable> runnable = runnables_[0];
			if (!runnable->isMulti())
			{
				throw std::runtime_error(runnable->getDescription() + " is for simple convolution, not multi!");
				return;
			}
			testCropped();
		}
		else if (runnables_.size() == 2)
		{
			auto runnable1 = runnables_[0];
			auto runnable2 = runnables_[1];
			if (!runnable1->isMulti())
			{
				throw std::runtime_error(runnable1->getDescription() + " is for simple convolution, not multi!");
				return;
			}
			if (!runnable2->isMulti())
			{
				throw std::runtime_error(runnable2->getDescription() + " is for simple convolution, not multi!");
				return;
			}
			testCropped();
		}
		else
		{
			throw std::runtime_error("Too many runables, maximum is 2");
		}
	}

	void TestMulti::testAlone(shared_ptr<Runnable> runnable, uint replications)
	{
		if (!runnable->isMulti())
		{
			std::cerr << runnable->getDescription() + " is not for multi nonvolution";
			return;
		}
		TestMultiBuilder builder;
		
			
			//.addFilterGroup(vector<shared_ptr<Filter>>({ Test::get3x3Filter(),Test::get3x3Filter(),Test::get3x3Filter() }))
			const int imageCount = 7;
			

			{
				vector<shared_ptr<Filter>> filterGroup;
				for (int i = 0; i < imageCount; i++)
				{
					filterGroup.push_back(Test::get3x3Filter());
				}
				builder.addFilterGroup(filterGroup);
			}
			{
				vector<shared_ptr<Filter>> filterGroup;
				for (int i = 0; i < imageCount; i++)
				{
					filterGroup.push_back(Test::get3x3Filter());
				}
				builder.addFilterGroup(filterGroup);
			}
			for (int i = 0; i < imageCount; i++)
			{
				builder.addImagePath(INPUT_IMAGE_PATHS[0]);
			}
			builder
			.addRunnable(runnable)
			.setReplications(replications);
		builder.build().testCropped();
	}

	void TestMulti::testAgainstEachOther(shared_ptr<Runnable> runnable1, shared_ptr<Runnable> runnable2, uint replications)
	{
		if (!runnable1->isMulti())
		{
			std::cerr << runnable1->getDescription() + " is not for multi nonvolution";
			return;
		}
		if (!runnable2->isMulti())
		{
			std::cerr << runnable2->getDescription() + " is not for multi nonvolution";
			return;
		}
		TestMultiBuilder builder;
		/*
		builder
			.addFilterGroup(vector<shared_ptr<Filter>>({ Test::get1x1Filter(),Test::get1x1Filter(),Test::get1x1Filter(), Test::get1x1Filter() }))
			.addFilterGroup(vector<shared_ptr<Filter>>({ Test::get3x3Filter(),Test::get3x3Filter(),Test::get3x3Filter(), Test::get3x3Filter() }))
			.addFilterGroup(vector<shared_ptr<Filter>>({ Test::get5x5Filter(),Test::get5x5Filter(),Test::get5x5Filter(), Test::get5x5Filter() }))
			.addFilterGroup(vector<shared_ptr<Filter>>({ Test::get7x7Filter(),Test::get7x7Filter(),Test::get7x7Filter(), Test::get7x7Filter() }))
			.addFilterGroup(vector<shared_ptr<Filter>>({ Test::get9x9Filter(),Test::get9x9Filter(),Test::get9x9Filter(), Test::get9x9Filter() }))
			.addFilterGroup(vector<shared_ptr<Filter>>({ Test::get11x11Filter(),Test::get11x11Filter(),Test::get11x11Filter(), Test::get11x11Filter() }))
			.addFilterGroup(vector<shared_ptr<Filter>>({ Test::get13x13Filter(),Test::get13x13Filter(),Test::get13x13Filter(), Test::get13x13Filter() }))
			.addFilterGroup(vector<shared_ptr<Filter>>({ Test::get15x15Filter(),Test::get15x15Filter(),Test::get15x15Filter(), Test::get15x15Filter() }))
			*/
		
			const int imageSize = 128;
				for (int i = 0; i < imageSize; i++)
				{
					builder.addImagePath(INPUT_IMAGE_PATHS[0]);
				}
				for (int k = 0; k < 50; k++)
				{
					{
						vector<shared_ptr<Filter>> fg;
						for (int i = 0; i < imageSize; i++)
						{
							fg.push_back(Test::get3x3Filter());
						}
						builder.addFilterGroup(fg);
					}
					{
						vector<shared_ptr<Filter>> fg;
						for (int i = 0; i < imageSize; i++)
						{
							fg.push_back(Test::get3x3Filter1());
						}
						builder.addFilterGroup(fg);
					}
					{
						vector<shared_ptr<Filter>> fg;
						for (int i = 0; i < imageSize; i++)
						{
							fg.push_back(Test::get3x3Filter3());
						}
						builder.addFilterGroup(fg);
					}
				}
		
			//.setImagePaths({ INPUT_IMAGE_PATH, INPUT_IMAGE_PATH, INPUT_IMAGE_PATH ,INPUT_IMAGE_PATH })
			builder
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


