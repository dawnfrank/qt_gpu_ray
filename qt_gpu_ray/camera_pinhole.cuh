#pragma once

#include "camera.cuh"

class Camera_Pinhole :public Camera {
public:
	__device__ Camera_Pinhole(Vec3 eye, Vec3 at, Vec3 up = Vec3(0, 1, 0))
		:Camera(eye, at, up)
	{
		angle = 0;
	}
	__device__ Vec3 ray_direction(const Vec2 p) const {
		Vec3 dir = p.x*coord.x + p.y*coord.y - d*coord.z;
		dir.normalize();
		return dir;
	}

	double d;
};