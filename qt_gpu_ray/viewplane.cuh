#pragma once

#include "sampler.cuh"

class ViewPlane {
public:
	__device__ void set_sampler(Sampler*sp) {
		if (sampler_ptr) {
			delete sampler_ptr;
			sampler_ptr = nullptr;
		}
		sampler_ptr = sp;
	}
	__device__ int get_samples_num() { return sampler_ptr->num_samples; }

	int hres;
	int vres;
	double pixel_size;
	double gamma;
	double inv_gamma;

	Sampler *sampler_ptr;
};