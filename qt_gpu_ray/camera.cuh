#pragma once

#include "mymath.cuh"

//right-hand coordinate
class Camera {
public:
	__device__ Camera(Vec3 eye, Vec3 at, Vec3 up = Vec3(0, 1, 0)) :
		eye(eye),
		at(at),
		up(up)
	{
		compute_coord();
	}
	__device__ void compute_coord() {
		Vec3 x, y, z;
		z = eye - at;
		z.normalize();
		y = Vec3(0, 1, 0);
		x = cross(y, z);
		x.normalize();
		y = cross(z, x);
		coord = Mat3(x, y, z);
	}

	__device__ void move(Vec3 dir) {
		Vec3 movePos;
		movePos.x = coord.x*dir;
		movePos.y = coord.y*dir;
		movePos.z = coord.z*dir;

		eye += movePos;
		at += movePos;
		compute_coord();
	}

	__device__ void rotate(Vec3 angle) {
		Vec3 cameraZ = eye - at;
		cameraZ = cameraZ*coord;
		eye += cameraZ;
		compute_coord();
	}

	__device__ void zoom(int z) {
		Vec3 movePos;
		movePos.x = z*coord.x.z;
		movePos.y = z*coord.y.z;
		movePos.z = z*coord.z.z;

		eye += movePos;
	}

	Vec3 eye, at, up;
	Mat3 coord;
	double exposure_time;
	double angle;
};