#pragma once


#include "mymath.cuh"
#include "ray.cuh"
#include "shaderec.cuh"

struct ObjectBase {
	__device__ virtual bool hit(const Ray& r, double& tmin, ShaderRec& rec) const = 0;
	__device__ virtual RGBColor get_color() {
		return color;
	}
	RGBColor color;
};