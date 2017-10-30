#pragma once

#include "objectbase.cuh"

class Sphere :public ObjectBase {
public:
	__device__ Sphere() {}
	__device__ Sphere(Vec3 cen, double r) : center(cen), radius(r) {};

	__device__ bool hit(const Ray& r, double &tmin, ShaderRec& rec) const;

	Vec3 center;
	double radius;
};