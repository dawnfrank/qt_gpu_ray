#pragma once

#include "drand48.cuh"
#include "vector.cuh"
#include "mymath.cuh"
#include "define.cuh"

class Sampler {
public:
	__device__ Sampler(const int num_samples = DEFAULT_SAMPLES, const int num_sets = DEFAULT_SETS) :
		num_samples(num_samples),
		num_sets(num_sets)
	{}

	__device__ virtual void generate_samples() = 0;
	__device__ Vec2 sample_unit_square() {
		if (count%num_samples == 0)jump = int(drand48()*num_sets)*num_samples;
		return samples[jump + count++ % num_samples];
	}
	__device__ void map_samples_to_unit_disk(void) {
		int size = samples.size();
		double r, phi;		// polar coordinates
		Vec2 sp; 		// sample point on unit disk

		disk_samples.reserve(size);

		for (int j = 0; j < size; j++) {
			// map sample point to [-1, 1] X [-1,1]

			sp.x = 2.0 * samples[j].x - 1.0;
			sp.y = 2.0 * samples[j].y - 1.0;

			if (sp.x > -sp.y) {			// sectors 1 and 2
				if (sp.x > sp.y) {		// sector 1
					r = sp.x;
					phi = sp.y / sp.x;
				}
				else {					// sector 2
					r = sp.y;
					phi = 2 - sp.x / sp.y;
				}
			}
			else {						// sectors 3 and 4
				if (sp.x < sp.y) {		// sector 3
					r = -sp.x;
					phi = 4 + sp.y / sp.x;
				}
				else {					// sector 4
					r = -sp.y;
					if (sp.y != 0.0)	// avoid division by zero at origin
						phi = 6 - sp.x / sp.y;
					else
						phi = 0.0;
				}
			}

			phi *= PI / 4.0;

			disk_samples[j].x = r * cos(phi);
			disk_samples[j].y = r * sin(phi);
		}

		samples.erase(samples.begin(), samples.end());
	}
	__device__ void map_samples_to_hemisphere(const double exp) {
		int size = samples.size();
		hemisphere_samples.reserve(num_samples * num_sets);

		for (int j = 0; j < size; j++) {
			double cos_phi = cos(2.0 * PI * samples[j].x);
			double sin_phi = sin(2.0 * PI * samples[j].x);
			double cos_theta = pow((1.0 - samples[j].y), 1.0 / (exp + 1.0));
			double sin_theta = sqrt(1.0 - cos_theta * cos_theta);
			double pu = sin_theta * cos_phi;
			double pv = sin_theta * sin_phi;
			double pw = cos_theta;
			hemisphere_samples.push_back(Vec3(pu, pv, pw));
		}
	}

	int num_samples;
	int num_sets;
	Vector<Vec2> samples;
	Vector<Vec2> disk_samples;
	Vector<Vec3> hemisphere_samples;
	unsigned long count = 0;
	int jump = 0;
};