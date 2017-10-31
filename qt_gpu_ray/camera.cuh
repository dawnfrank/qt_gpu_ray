#pragma once

#include "mymath.cuh"

class Camera {
public:
	__device__ Camera(Vec3 eye, Vec3 at, Vec3 up = Vec3(0, 1, 0)) :eye(eye), at(at), up(up) {}
	__device__ void compute_uvw() {
		w = eye - at;
		w.normalize();
		u = cross(w, up);
		u.normalize();
		v = cross(w, u);
	}

	Vec3 eye, at, up;
	Vec3 u, v, w;
	double exposure_time;
	double angle;
};