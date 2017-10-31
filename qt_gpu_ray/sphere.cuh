#pragma once

#include "define.cuh"
#include "objectbase.cuh"

class Sphere :public ObjectBase {
public:
	__device__ Sphere() {}
	__device__ Sphere(Vec3 cen, double r) : center(cen), radius(r) {};

	__device__ bool hit(const Ray& r, double &tmin, ShaderRec& rec) const {
		Vec3 oc = r.origin - center;
		double a = r.direction*r.direction;
		double b = oc*r.direction;
		double c = oc*oc - radius*radius;
		double discriminant = b*b - a*c;


		if (discriminant > 0) {
			double temp = (-b - sqrt(discriminant)) / a;
			if (temp > kEpsilon) {
				tmin = temp;
				rec.hit_point = oc + temp*r.direction;
				rec.normal = (oc + temp*r.direction) / radius;
				return true;
			}
			temp = (-b + sqrt(discriminant)) / a;
			if (temp > kEpsilon) {
				tmin = temp;
				rec.hit_point = oc + temp*r.direction;
				rec.normal = (oc + temp*r.direction) / radius;
				return true;
			}
		}
		return false;
	}

	Vec3 center;
	double radius;
};