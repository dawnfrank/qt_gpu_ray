
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "define.cuh"

extern "C" cudaError_t InitCuda(const int w, const int h, unsigned char **dev_bitmap);
extern "C" cudaError_t CalculateCuda(const int w, const int h, unsigned char *dev_bitmap,unsigned char *host_bitmap);


__global__ void Raykernel(const int w, const int h, unsigned char *dev_bitmap)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x < w && y < h) {
		int offset = x + y * w;
		dev_bitmap[offset * 4] = 255;
		dev_bitmap[offset * 4 + 1] = 0;
		dev_bitmap[offset * 4 + 2] = 0;
		dev_bitmap[offset * 4 + 3] = 255;
	}
}

cudaError_t InitCuda(const int w, const int h, unsigned char **dev_bitmap)
{
	const int imageSize = w*h * 4;
	cudaError_t cudaStatus = cudaMalloc(dev_bitmap, imageSize * sizeof(int));
	return cudaStatus;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t CalculateCuda(const int w, const int h, unsigned char *dev_bitmap, unsigned char *host_bitmap)
{
	const int imageSize = w*h * 4;
	dim3 blocks((w + DIM - 1) / DIM, (h + DIM - 1) / DIM);
	dim3 threads(DIM, DIM);
	// Launch a kernel on the GPU with one thread for each element.
	Raykernel << <blocks, threads >> > (w, h, dev_bitmap);

	cudaError_t cudaStatus = cudaMemcpy(host_bitmap, dev_bitmap, imageSize, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
	}
	return cudaStatus;
}
