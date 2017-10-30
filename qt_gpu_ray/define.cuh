#pragma once

#include <stdlib.h>
#include "rgbcolor.cuh"
#include "device_launch_parameters.h"

#define DIM 32
#define DEFAULT_VECTOR_SIZE 10

__device__ const int DEFAULT_SAMPLES = 16;
__device__ const int DEFAULT_SETS = 53;

__device__ const double 	TWO_PI = 6.2831853071795864769;
__device__ const double 	PI_ON_180 = 0.0174532925199432957;
__device__ const double 	invPI = 0.3183098861837906715;
__device__ const double 	invTWO_PI = 0.1591549430918953358;

__device__ const double 	kEpsilon = 0.0001;
__device__ const double		kHugeValue = 1.0E10;
__device__ const double 	invRAND_MAX = 1.0 / (float)RAND_MAX;

const RGBColor	black(0.0, 0.0, 0.0);
const RGBColor	white(1.0, 0.0, 0.0);
const RGBColor	red(1.0, 0.0, 0.0);