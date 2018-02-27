#include "ThreadPool.h"



ThreadPool::ThreadPool(int c) :
	threadCount_(c),
	end_(false),
	undoneTasks_(0),
	destroyedThreads_(0)
{
	if (c < 1)
	{
		throw std::runtime_error("Number of threads must be bigger than 0");
	}
	for (int i = 0; i < c; i++)
	{
		std::thread([this] { this->run(); }).detach();
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
		mutex_.unlock();
		++undoneTasks_;
		conditionVariable_.notify_one();
	}
}

void ThreadPool::finishAll()
{
	std::unique_lock<std::mutex> lock(mutex_);
	while (undoneTasks_ > 0)
	{
		conditionVariable_.wait(lock);
	}
}

void ThreadPool::run()
{
	std::function<void(void)> run;
	while (!end_) {
		std::unique_lock<std::mutex> lock(mutex_);
		while (undoneTasks_ == 0 && !end_)
		{
			conditionVariable_.notify_one();
			conditionVariable_.wait(lock);
		}
		if (!tasks_.empty()) {
			run = std::move(tasks_.front());
			tasks_.pop();
			lock.unlock();
			run();
			--undoneTasks_;
		}
	}
	
	++destroyedThreads_;
	conditionVariable_.notify_one();
}

void ThreadPool::joinAll()
{
	if (!end_)
	{
		std::unique_lock<std::mutex> lock(mutex_);
		conditionVariable_.wait(lock, [&undoneTasks = undoneTasks_]() -> bool { return undoneTasks == 0; });
		end_ = true;
		conditionVariable_.notify_all();
		conditionVariable_.wait(lock, [&threadCount = threadCount_, &destroyedThreads = destroyedThreads_]() { return destroyedThreads == threadCount; });
	}
}
