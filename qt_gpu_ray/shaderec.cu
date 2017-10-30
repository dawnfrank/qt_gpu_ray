#include "shaderec.cuh"

ShaderRec::ShaderRec() :
	hit_an_object(false),
	normal(),
	hit_point(),
	color()
{}

/*
ShaderRec::ShaderRec(const ShaderRec &sr) :
	hit_an_object(sr.hit_an_object),
	normal(sr.normal),
	hit_point(sr.hit_point),
	color(sr.color)
{}
*/