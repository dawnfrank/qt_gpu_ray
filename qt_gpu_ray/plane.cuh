#pragma once

#include "define.cuh"
#include "objectbase.cuh"

struct Plane :public ObjectBase {
	__device__ Plane() {}
	__device__ Plane(Vec3 p, Vec3 norm) : point(p), normal(norm) {};

	__device__ virtual bool hit(const Ray& r, double& tmin, ShaderRec& rec) const {
		double t = (point - r.origin)*normal / (r.direction*normal);
		if (t>kEpsilon) {
			tmin = t;
			rec.normal = normal;
			rec.hit_point = r.origin + t*r.direction;
			return true;
		}
		return false;
	}

private:
	Vec3 point;
	Vec3 normal;
};