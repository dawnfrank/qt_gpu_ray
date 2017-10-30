#pragma once

#include "device_launch_parameters.h"

#define drand_m 0x100000000LL  
#define drand_c 0xB16  
#define drand_a 0x5DEECE66DLL  

__device__ static unsigned long long seed = 1;

__device__ double drand48(void) {
	seed = (drand_a * seed + drand_c) & 0xFFFFFFFFFFFFLL;
	unsigned long long x = seed >> 16;
	return  ((double)x / (double)drand_m);
}
