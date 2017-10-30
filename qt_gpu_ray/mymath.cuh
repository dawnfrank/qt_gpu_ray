#pragma once

#include <math.h>
#include "device_launch_parameters.h"

#define EPSILON 1e-6f
#define PI 3.14159265358979323846f
#define DEG2RAD (PI / 180.0f)
#define RAD2DEG (180.0f / PI)

struct Vec2;
struct Vec3;

/*****************************************************************************/
/*                                                                           */
/* Vec2                                                                      */
/*                                                                           */
/*****************************************************************************/

struct Vec2 {
	__device__ inline Vec2() : x(0), y(0) { }
	__device__ inline Vec2(double x, double y) : x(x), y(y) { }
	__device__ inline Vec2(const double *v) : x(v[0]), y(v[1]) { }
	__device__ inline Vec2(const Vec2 &v) : x(v.x), y(v.y) { }

	__device__ inline int operator==(const Vec2 &v) { return (fabs(x - v.x) < EPSILON && fabs(y - v.y) < EPSILON); }
	__device__ inline int operator!=(const Vec2 &v) { return !(*this == v); }

	__device__ inline const Vec2 operator*(double f) const { return Vec2(x * f, y * f); }
	__device__ inline const Vec2 operator/(double f) const { return Vec2(x / f, y / f); }
	__device__ inline const Vec2 operator+(const Vec2 &v) const { return Vec2(x + v.x, y + v.y); }
	__device__ inline const Vec2 operator-() const { return Vec2(-x, -y); }
	__device__ inline const Vec2 operator-(const Vec2 &v) const { return Vec2(x - v.x, y - v.y); }

	__device__ inline Vec2 &operator*=(double f) { return *this = *this * f; }
	__device__ inline Vec2 &operator/=(double f) { return *this = *this / f; }
	__device__ inline Vec2 &operator+=(const Vec2 &v) { return *this = *this + v; }
	__device__ inline Vec2 &operator-=(const Vec2 &v) { return *this = *this - v; }

	__device__ inline double operator*(const Vec2 &v) const { return x * v.x + y * v.y; }

	__device__ inline operator double*() { return (double*)&x; }
	__device__ inline operator const double*() const { return (double*)&x; }

	__device__ inline double &operator[](int i) { return ((double*)&x)[i]; }
	__device__ inline const double operator[](int i) const { return ((double*)&x)[i]; }

	__device__ inline double magnitude() const { return sqrt(x * x + y * y); }
	__device__ inline double normalize() {
		double inv, length = magnitude();
		if (length < EPSILON) return 0.0;
		inv = 1.0 / length;
		x *= inv;
		y *= inv;
		return length;
	}

	union {
		struct {
			double x, y;
		};
		double v[2];
	};
};

__device__ inline Vec2 operator*(double t, const Vec2 &v) { return v * t; }
__device__ inline Vec2 operator/(double t, const Vec2 &v) { return v / t; }


/*****************************************************************************/
/*                                                                           */
/* Vec3                                                                      */
/*                                                                           */
/*****************************************************************************/

struct Vec3 {
	__device__ inline Vec3() : x(0), y(0), z(0) { }
	__device__ inline Vec3(double x, double y, double z) : x(x), y(y), z(z) { }
	__device__ inline Vec3(const double *v) : x(v[0]), y(v[1]), z(v[2]) { }
	__device__ inline Vec3(const Vec3 &v) : x(v.x), y(v.y), z(v.z) { }

	__device__ inline int operator==(const Vec3 &v) { return (fabs(x - v.x) < EPSILON && fabs(y - v.y) < EPSILON && fabs(z - v.z) < EPSILON); }
	__device__ inline int operator!=(const Vec3 &v) { return !(*this == v); }

	__device__ inline const Vec3 operator*(double f) const { return Vec3(x * f, y * f, z * f); }
	__device__ inline const Vec3 operator/(double f) const { return Vec3(x / f, y / f, z / f); }
	__device__ inline const Vec3 operator+(const Vec3 &v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
	__device__ inline const Vec3 operator-() const { return Vec3(-x, -y, -z); }
	__device__ inline const Vec3 operator-(const Vec3 &v) const { return Vec3(x - v.x, y - v.y, z - v.z); }

	__device__ inline Vec3 &operator*=(double f) { return *this = *this * f; }
	__device__ inline Vec3 &operator/=(double f) { return *this = *this / f; }
	__device__ inline Vec3 &operator+=(const Vec3 &v) { return *this = *this + v; }
	__device__ inline Vec3 &operator-=(const Vec3 &v) { return *this = *this - v; }

	__device__ inline double operator*(const Vec3 &v) const { return x * v.x + y * v.y + z * v.z; }

	__device__ inline operator double*() { return (double*)&x; }
	__device__ inline operator const double*() const { return (double*)&x; }

	__device__ inline double &operator[](int i) { return ((double*)&x)[i]; }
	__device__ inline const double operator[](int i) const { return ((double*)&x)[i]; }

	__device__ inline double magnitude() const { return sqrt(x * x + y * y + z * z); }
	__device__ inline double normalize() {
		double inv, length = magnitude();
		if (length < EPSILON) return 0.0;
		inv = 1.0 / length;
		x *= inv;
		y *= inv;
		z *= inv;
		return length;
	}
	__device__ inline void cross(const Vec3 &v1, const Vec3 &v2) {
		x = v1.y * v2.z - v1.z * v2.y;
		y = v1.z * v2.x - v1.x * v2.z;
		z = v1.x * v2.y - v1.y * v2.x;
	}

	union {
		struct {
			double x, y, z;
		};
		double v[3];
	};
};

__device__ inline Vec3 cross(const Vec3 &v1, const Vec3 &v2) {
	Vec3 ret;
	ret.x = v1.y * v2.z - v1.z * v2.y;
	ret.y = v1.z * v2.x - v1.x * v2.z;
	ret.z = v1.x * v2.y - v1.y * v2.x;
	return ret;
}

__device__ inline Vec3 operator*(double t, const Vec3 &v) { return v * t; }
__device__ inline Vec3 operator/(double t, const Vec3 &v) { return v / t; }