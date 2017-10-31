#pragma once

#include "camera.cuh"

class Camera_Pinhole :public Camera {
public:
	__device__ Camera_Pinhole(Vec3 eye, Vec3 at, Vec3 up = Vec3(0, 1, 0))
		:Camera(eye, at, up)
	{
		zoom = 0.5;
		d = at[2] - eye[2];
		angle = 0;
	}
	__device__ Vec3 ray_direction(const Vec2 p) const {
		Vec3 dir = p.x*u + p.y*v - d*w;
		dir.normalize();
		return dir;
	}

	double d;
	double zoom;
};