
#include "sphere.cuh"
#include "define.cuh"


bool Sphere::hit(const Ray& r, double& tmin, ShaderRec& rec) const {
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