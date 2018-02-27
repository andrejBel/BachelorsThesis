#ifndef ThreadPool_H
#define ThreadPool_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <queue>
#include <mutex>
#include <thread>
#include <iostream>

class ThreadPool {

private:

	std::queue<std::function<void(void)> > tasks_;
	bool end_;
	std::condition_variable conditionVariable_;
	std::mutex mutex_;
	std::atomic<int> destroyedThreads_;
	std::atomic<int> undoneTasks_;
	int threadCount_;
	void joinAll();
	void run();
	
	
	
public:


	ThreadPool(int c);

	ThreadPool(const ThreadPool& other) = delete;

	ThreadPool& operator=(const ThreadPool& other) = delete;

	~ThreadPool();

	void addTask(std::function<void(void)> job);

	void finishAll();

};

#endif /* ThreadPool_H */