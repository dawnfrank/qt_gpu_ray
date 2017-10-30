#pragma once

#include "mymath.cuh"

struct Ray {
	__device__ Ray() {};
	__device__ Ray(const Vec3& ori, const Vec3& dir) :origin(ori), direction(dir) { direction.normalize(); }
	__device__ Ray(const Ray& r) { origin = r.origin; direction = r.direction; }

	__device__ Ray& operator=(const Ray&r) { origin = r.origin; direction = r.direction; return *this; };

	__device__ Vec3 point_at_parameter(double t) const { return origin + direction*t; }

	Vec3 origin;
	Vec3 direction;
};