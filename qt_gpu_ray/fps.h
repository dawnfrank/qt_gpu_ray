#pragma once

#include <chrono>
#include <iostream>
#include <time.h>

class FPS {
public:
	void Update() {
		++_frame_counter;
		if (_last_time + std::chrono::seconds(1) < _timer.now()) {
			_last_time = _timer.now();
			_fps = _frame_counter;
			_frame_counter = 0;
			std::cout << "FPS:" << _fps << std::endl;
		}

//		std::cout << "Time: " << clock() - _last_clock << "ms" <<std::endl;
		_last_clock = clock();
	};
private:
	std::chrono::steady_clock _timer = std::chrono::steady_clock();
	std::chrono::time_point<std::chrono::steady_clock> _last_time;
	uint64_t _frame_counter = 0;
	uint64_t _fps = 0;

	clock_t _last_clock = clock();
};