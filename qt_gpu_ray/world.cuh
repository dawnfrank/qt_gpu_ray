#pragma once

#include "sphere.cuh"
#include "viewplane.cuh"
#include "objectbase.cuh"
#include "camera.cuh"

class World {
public:
	__device__ World() {}
	__device__ World(const World& w) {}

	__device__ inline void add_object(ObjectBase* object_ptr) { objects.push_back(object_ptr); }
	__device__ ShaderRec hit_objects(const Ray& ray) {
		ShaderRec sr;

		double t;
		double tmin = kHugeValue;
		int num_objects = objects.size();

		for (int i = 0; i < num_objects; ++i) {
			if (objects[i]->hit(ray, t, sr) && (t < tmin)) {
				sr.hit_an_object = true;
				sr.color = objects[i]->get_color();
				tmin = t;
			}
		}
		if (sr.hit_an_object)return sr;
		t = 0.5*(ray.direction.y / vp.vres + 1.0);
		sr.color = (1.0 - t)*RGBColor(1.0, 1.0, 1.0) + t*RGBColor(0.5, 0.7, 1.0);
		return sr;
	}

	Sphere* sp_ptr;
	ViewPlane vp;
	Camera* camera_ptr;
	Vector<ObjectBase*> objects;
};