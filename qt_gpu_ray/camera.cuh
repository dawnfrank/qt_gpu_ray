#pragma once

#include "mymath.cuh"

//right-hand coordinate
class Camera {
public:
	__device__ Camera(Vec3 eye, Vec3 at, Vec3 up = Vec3(0, 1, 0)) :eye(eye), at(at), up(up) {}
	__device__ void compute_coord() {
		Vec3 x, y, z;
		z = eye - at;
		z.normalize();
		y = (0, 1, 0);
		x = cross(y, z);
		x.normalize();
		y = cross(z, x);
		coord = Mat3(x, y, z);
	}

	Vec3 eye, at, up;
	Mat3 coord;
	double exposure_time;
	double angle;
};