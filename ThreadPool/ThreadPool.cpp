#include "ThreadPool.h"


ThreadPool::ThreadPool(int c):
	end_(false),
	undoneTasks_(0)
{
	if (c < 1)
	{
		throw std::runtime_error("Number of threads must be bigger than 0");
	}
	threads_.resize(c);
	for (int i = 0; i < c; i++)
	{
		threads_[i] = std::move(std::thread(&ThreadPool::run, this));
	}
}

ThreadPool::~ThreadPool() 
{
	joinAll();
}

void ThreadPool::addTask(std::function<void(void)> job)
{
	if (!end_) {
		mutex_.lock();
		tasks_.push(job);
		++undoneTasks_;
		mutex_.unlock();
		conditionVariable_.notify_one();
	}
}

void ThreadPool::finishAll()
{
	std::unique_lock<std::mutex> lock(mutex_);
	conditionVariable_.wait(lock, [&undoneTasks_ = undoneTasks_]() { return undoneTasks_ == 0; });
}

void ThreadPool::run()
{
	std::function<void(void)> run;
	while (true) {
		std::unique_lock<std::mutex> lock(mutex_);
		ThreadPool::Action nextAction = getNextAction();
		if (nextAction == ThreadPool::Action::SLEEP)
		{
			conditionVariable_.wait(lock, [this]() { return getNextAction() != ThreadPool::Action::SLEEP; });
			nextAction = getNextAction();
		}
		if (nextAction == ThreadPool::Action::END) {
			break;
		}
		else if (nextAction == ThreadPool::Action::WORK)
		{
			if (!tasks_.empty()) {
				run = std::move(tasks_.front());
				tasks_.pop();
				lock.unlock();
				run();
				--undoneTasks_;
				conditionVariable_.notify_all();
			}
		}
		
	}
}

ThreadPool::Action ThreadPool::getNextAction()
{
	if (undoneTasks_ > 0)
	{
		return ThreadPool::Action::WORK;
	}
	if (end_) 
	{
		return ThreadPool::Action::END;
	}
	return ThreadPool::Action::SLEEP;
}

void ThreadPool::joinAll()
{
	if (!end_)
	{
		std::unique_lock<std::mutex> lock(mutex_);
		conditionVariable_.wait(lock, [&undoneTasks = undoneTasks_]() -> bool { return undoneTasks == 0; });
		end_ = true;
		lock.unlock();
		conditionVariable_.notify_all();
		for (std::thread& thread  :threads_)
		{
			thread.join();
		}
	}
}
