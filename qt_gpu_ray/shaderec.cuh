#pragma once

#include "rgbcolor.cuh"
#include "mymath.cuh"

class ShaderRec {
public:
	__device__ ShaderRec() :
		hit_an_object(false),
		normal(),
		hit_point(),
		color()
	{}
	__device__ ShaderRec(const ShaderRec &sr) :
		hit_an_object(sr.hit_an_object),
		normal(sr.normal),
		hit_point(sr.hit_point),
		color(sr.color)
	{}

	Vec3 normal;
	Vec3 hit_point;
	bool hit_an_object;
	RGBColor color;
};