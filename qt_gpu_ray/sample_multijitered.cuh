#pragma once

#include "sampler.cuh"

class Sample_MultiJittered :public Sampler {
public:
	__device__ Sample_MultiJittered(const int num_samples = DEFAULT_SAMPLES, const int num_sets = DEFAULT_SETS)
		:Sampler(num_samples, num_sets)
	{
		generate_samples();
	}

	__device__ void generate_samples() {
		Vector<int> gridNums;
		Vector<int> blockNums;
		int gridIndex, gridNum, blockIndex, blockNum;
		int gridRow, gridCol, blockRow, blockCol;
		int n = (int)sqrtf(num_samples);
		double u, v;

		for (int k = 0; k != num_sets; ++k)
		{
			gridNums.clear();
			blockNums.clear();
			for (int num = 0; num != num_samples; ++num) {
				gridNums.push_back(num);
				blockNums.push_back(num);
			}
			for (int i = 0; i != num_samples; ++i) {
				gridIndex = int(drand48()*gridNums.size());
				gridNum = gridNums[gridIndex];
				gridNums.erase(gridNums.begin() + gridIndex);
				gridRow = gridNum % n;
				gridCol = gridNum / n;

				blockIndex = int(drand48()*blockNums.size());
				blockNum = blockNums[blockIndex];
				blockNums.erase(blockNums.begin() + blockIndex);
				blockRow = blockNum % n;
				blockCol = blockNum / n;

				u = double(gridRow*n + blockRow + drand48()) / double(num_samples);
				v = double(gridCol*n + blockCol + drand48()) / double(num_samples);

				samples.push_back(Vec2(u, v));
			}
		}
	}
};