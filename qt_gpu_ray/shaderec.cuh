#pragma once

#include "rgbcolor.cuh"
#include "mymath.cuh"

class ShaderRec {
public:
	__device__ ShaderRec();
//	__device__ ShaderRec(const ShaderRec &sr);

	Vec3 normal;
	Vec3 hit_point;
	bool hit_an_object;
	RGBColor color;
};