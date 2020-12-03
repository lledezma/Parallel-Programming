#ifndef STOP_WATCH_H
#include <chrono>
class StopWatch
{
public:
	using TimePoint = std::chrono::time_point<std::chrono::steady_clock>;

	StopWatch() {

	}

	~StopWatch() {

	}

	void start() {
		this->begin = std::chrono::high_resolution_clock::now();
	}
	void stop() {
		this->end = std::chrono::high_resolution_clock::now();
	}

	double elapsedTime() {
		return std::chrono::duration<double, std::milli>(this->end - this->begin).count();
	}

private:
	TimePoint begin;
	TimePoint end;

};

#endif // !STOP_WATCH_H

