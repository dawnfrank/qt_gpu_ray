#pragma once

#include <math.h>
#include "device_launch_parameters.h"

#define EPSILON 1e-6f
#define PI 3.14159265358979323846f
#define DEG2RAD (PI / 180.0f)
#define RAD2DEG (180.0f / PI)

struct Vec2;
struct Vec3;
struct Mat3;

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

	__device__ Vec3 operator*(const Mat3 &mat) const;

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


/*****************************************************************************/
/*                                                                           */
/* Mat3                                                                      */
/*                                                                           */
/*****************************************************************************/

struct Mat3 {
	__device__ Mat3() {
		mat[0] = 1.0; mat[3] = 0.0; mat[6] = 0.0;
		mat[1] = 0.0; mat[4] = 1.0; mat[7] = 0.0;
		mat[2] = 0.0; mat[5] = 0.0; mat[8] = 1.0;
	}
	__device__ Mat3(const double *m) {
		mat[0] = m[0]; mat[3] = m[3]; mat[6] = m[6];
		mat[1] = m[1]; mat[4] = m[4]; mat[7] = m[7];
		mat[2] = m[2]; mat[5] = m[5]; mat[8] = m[8];
	}
	__device__ Mat3(const Mat3 &m) {
		mat[0] = m[0]; mat[3] = m[3]; mat[6] = m[6];
		mat[1] = m[1]; mat[4] = m[4]; mat[7] = m[7];
		mat[2] = m[2]; mat[5] = m[5]; mat[8] = m[8];
	}
	__device__ Mat3(const Vec3 &v1, const Vec3 &v2, const Vec3 &v3) {
		mat[0] = v1[0]; mat[3] = v1[1]; mat[6] = v1[2];
		mat[1] = v2[0]; mat[4] = v2[1]; mat[7] = v2[2];
		mat[2] = v3[0]; mat[5] = v3[1]; mat[8] = v3[2];
	}
	__device__ Mat3 operator*(double f) const {
		Mat3 ret;
		ret[0] = mat[0] * f; ret[3] = mat[3] * f; ret[6] = mat[6] * f;
		ret[1] = mat[1] * f; ret[4] = mat[4] * f; ret[7] = mat[7] * f;
		ret[2] = mat[2] * f; ret[5] = mat[5] * f; ret[8] = mat[8] * f;
		return ret;
	}
	__device__ Mat3 operator*(const Mat3 &m) const {
		Mat3 ret;
		ret[0] = mat[0] * m[0] + mat[3] * m[1] + mat[6] * m[2];
		ret[1] = mat[1] * m[0] + mat[4] * m[1] + mat[7] * m[2];
		ret[2] = mat[2] * m[0] + mat[5] * m[1] + mat[8] * m[2];
		ret[3] = mat[0] * m[3] + mat[3] * m[4] + mat[6] * m[5];
		ret[4] = mat[1] * m[3] + mat[4] * m[4] + mat[7] * m[5];
		ret[5] = mat[2] * m[3] + mat[5] * m[4] + mat[8] * m[5];
		ret[6] = mat[0] * m[6] + mat[3] * m[7] + mat[6] * m[8];
		ret[7] = mat[1] * m[6] + mat[4] * m[7] + mat[7] * m[8];
		ret[8] = mat[2] * m[6] + mat[5] * m[7] + mat[8] * m[8];
		return ret;
	}
	__device__ Mat3 operator+(const Mat3 &m) const {
		Mat3 ret;
		ret[0] = mat[0] + m[0]; ret[3] = mat[3] + m[3]; ret[6] = mat[6] + m[6];
		ret[1] = mat[1] + m[1]; ret[4] = mat[4] + m[4]; ret[7] = mat[7] + m[7];
		ret[2] = mat[2] + m[2]; ret[5] = mat[5] + m[5]; ret[8] = mat[8] + m[8];
		return ret;
	}
	__device__ Mat3 operator-(const Mat3 &m) const {
		Mat3 ret;
		ret[0] = mat[0] - m[0]; ret[3] = mat[3] - m[3]; ret[6] = mat[6] - m[6];
		ret[1] = mat[1] - m[1]; ret[4] = mat[4] - m[4]; ret[7] = mat[7] - m[7];
		ret[2] = mat[2] - m[2]; ret[5] = mat[5] - m[5]; ret[8] = mat[8] - m[8];
		return ret;
	}

	__device__ Mat3 &operator*=(double f) { return *this = *this * f; }
	__device__ Mat3 &operator*=(const Mat3 &m) { return *this = *this * m; }
	__device__ Mat3 &operator+=(const Mat3 &m) { return *this = *this + m; }
	__device__ Mat3 &operator-=(const Mat3 &m) { return *this = *this - m; }

	__device__ operator double*() { return mat; }
	__device__ operator const double*() const { return mat; }

	__device__ double &operator[](int i) { return mat[i]; }
	__device__ const double operator[](int i) const { return mat[i]; }

	__device__ Mat3 transpose() const {
		Mat3 ret;
		ret[0] = mat[0]; ret[3] = mat[1]; ret[6] = mat[2];
		ret[1] = mat[3]; ret[4] = mat[4]; ret[7] = mat[5];
		ret[2] = mat[6]; ret[5] = mat[7]; ret[8] = mat[8];
		return ret;
	}
	__device__ double det() const {
		double det;
		det = mat[0] * mat[4] * mat[8];
		det += mat[3] * mat[7] * mat[2];
		det += mat[6] * mat[1] * mat[5];
		det -= mat[6] * mat[4] * mat[2];
		det -= mat[3] * mat[1] * mat[8];
		det -= mat[0] * mat[7] * mat[5];
		return det;
	}
	__device__ Mat3 inverse() const {
		Mat3 ret;
		double idet = 1.0 / det();
		ret[0] = (mat[4] * mat[8] - mat[7] * mat[5]) * idet;
		ret[1] = -(mat[1] * mat[8] - mat[7] * mat[2]) * idet;
		ret[2] = (mat[1] * mat[5] - mat[4] * mat[2]) * idet;
		ret[3] = -(mat[3] * mat[8] - mat[6] * mat[5]) * idet;
		ret[4] = (mat[0] * mat[8] - mat[6] * mat[2]) * idet;
		ret[5] = -(mat[0] * mat[5] - mat[3] * mat[2]) * idet;
		ret[6] = (mat[3] * mat[7] - mat[6] * mat[4]) * idet;
		ret[7] = -(mat[0] * mat[7] - mat[6] * mat[1]) * idet;
		ret[8] = (mat[0] * mat[4] - mat[3] * mat[1]) * idet;
		return ret;
	}

	__device__ void zero() {
		mat[0] = 0.0; mat[3] = 0.0; mat[6] = 0.0;
		mat[1] = 0.0; mat[4] = 0.0; mat[7] = 0.0;
		mat[2] = 0.0; mat[5] = 0.0; mat[8] = 0.0;
	}
	__device__ void identity() {
		mat[0] = 1.0; mat[3] = 0.0; mat[6] = 0.0;
		mat[1] = 0.0; mat[4] = 1.0; mat[7] = 0.0;
		mat[2] = 0.0; mat[5] = 0.0; mat[8] = 1.0;
	}
	/*
	__device__ void rotate(const Vec3 &axis, double angle) {
		double rad = angle * DEG2RAD;
		double c = cos(rad);
		double s = sin(rad);
		Vec3 v = axis;
		v.normalize();
		double xx = v.x * v.x;
		double yy = v.y * v.y;
		double zz = v.z * v.z;
		double xy = v.x * v.y;
		double yz = v.y * v.z;
		double zx = v.z * v.x;
		double xs = v.x * s;
		double ys = v.y * s;
		double zs = v.z * s;
		mat[0] = (1.0 - c) * xx + c; mat[3] = (1.0 - c) * xy - zs; mat[6] = (1.0 - c) * zx + ys;
		mat[1] = (1.0 - c) * xy + zs; mat[4] = (1.0 - c) * yy + c; mat[7] = (1.0 - c) * yz - xs;
		mat[2] = (1.0 - c) * zx - ys; mat[5] = (1.0 - c) * yz + xs; mat[8] = (1.0 - c) * zz + c;
	}
	__device__ void rotate(double x, double y, double z, double angle) {
		rotate(Vec3(x, y, z), angle);
	}*/
	__device__ void rotate_x(double angle) {
		double rad = angle * DEG2RAD;
		double c = cos(rad);
		double s = sin(rad);
		mat[0] = 1.0; mat[3] = 0.0; mat[6] = 0.0;
		mat[1] = 0.0; mat[4] = c; mat[7] = -s;
		mat[2] = 0.0; mat[5] = s; mat[8] = c;
	}
	__device__ void rotate_y(double angle) {
		double rad = angle * DEG2RAD;
		double c = cos(rad);
		double s = sin(rad);
		mat[0] = c; mat[3] = 0.0; mat[6] = s;
		mat[1] = 0.0; mat[4] = 1.0; mat[7] = 0.0;
		mat[2] = -s; mat[5] = 0.0; mat[8] = c;
	}
	__device__ void rotate_z(double angle) {
		double rad = angle * DEG2RAD;
		double c = cos(rad);
		double s = sin(rad);
		mat[0] = c; mat[3] = -s; mat[6] = 0.0;
		mat[1] = s; mat[4] = c; mat[7] = 0.0;
		mat[2] = 0.0; mat[5] = 0.0; mat[8] = 1.0;
	}
	__device__ void scale(const Vec3 &v) {
		mat[0] = v.x; mat[3] = 0.0; mat[6] = 0.0;
		mat[1] = 0.0; mat[4] = v.y; mat[7] = 0.0;
		mat[2] = 0.0; mat[5] = 0.0; mat[8] = v.z;
	}
	__device__ void scale(double x, double y, double z) {
		scale(Vec3(x, y, z));
	}
	/*
	__device__ void orthonormalize() {
		Vec3 x(mat[0], mat[1], mat[2]);
		Vec3 y(mat[3], mat[4], mat[5]);
		Vec3 z;
		x.normalize();
		z.cross(x, y);
		z.normalize();
		y.cross(z, x);
		y.normalize();
		mat[0] = x.x; mat[3] = y.x; mat[6] = z.x;
		mat[1] = x.y; mat[4] = y.y; mat[7] = z.y;
		mat[2] = x.z; mat[5] = y.z; mat[8] = z.z;
	}*/

	union {
		struct {
			Vec3 x;
			Vec3 y;
			Vec3 z;
		};
		double mat[9];
	};
};

__device__ Vec3 Vec3::operator*(const Mat3 &mat) const {
	Vec3 ret;
	ret[0] = mat[0] * v[0] + mat[1] * v[1] + mat[2] * v[2];
	ret[1] = mat[3] * v[0] + mat[4] * v[1] + mat[5] * v[2];
	ret[2] = mat[6] * v[0] + mat[7] * v[1] + mat[8] * v[2];
	return ret;
}