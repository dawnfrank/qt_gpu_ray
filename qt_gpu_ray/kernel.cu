
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "sphere.cuh"
#include "plane.cuh"
#include "world.cuh"
#include "camera_pinhole.cuh"
#include "sample_multijitered.cuh"

extern "C" World* InitWorld(const int w, const int h);
extern "C" void RenderWorld(const int w, const int h, unsigned char *dev_bitmap, unsigned char *host_bitmap, World* dev_world);


__global__ void Raykernel(const int w, const int h, unsigned char *dev_bitmap, World* dev_world)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x < w && y < h) {
		int offset = x + y * w;

		RGBColor pixel_color;
		ShaderRec sr;
		Ray ray;
		Vec2 sp, pp;

		double distanceZ = (dev_world->camera_ptr->eye - dev_world->camera_ptr->at).magnitude();
		double pixel_size = dev_world->vp.pixel_size;
		int sample_num = dev_world->vp.get_samples_num();
		ray.origin = dev_world->camera_ptr->eye;

		for (int s = 0; s != sample_num; ++s) {
			sp = dev_world->vp.sampler_ptr->sample_unit_square();
			pp.x = pixel_size*(x - 0.5*(w - 1) + sp.x);
			pp.y = -pixel_size*(y - 0.5*(h - 1) + sp.y);
			ray.direction = Vec3(pp.x, pp.y, -distanceZ);

			sr = dev_world->hit_objects(ray);
			if (sr.hit_an_object) pixel_color += sr.color;
			else pixel_color += dev_world->bg_color;
		}
		pixel_color /= sample_num;
		

		dev_bitmap[offset * 4] = int(pixel_color[2] * 255.99);
		dev_bitmap[offset * 4 + 1] = int(pixel_color[1] * 255.99);
		dev_bitmap[offset * 4 + 2] = int(pixel_color[0] * 255.99);
		dev_bitmap[offset * 4 + 3] = 255;
	}
}

void RenderWorld(const int w, const int h, unsigned char *dev_bitmap, unsigned char *host_bitmap, World* dev_world)
{
	const int imageSize = w*h * 4;
	dim3 blocks((w + DIM - 1) / DIM, (h + DIM - 1) / DIM);
	dim3 threads(DIM, DIM);
	// Launch a kernel on the GPU with one thread for each element.
	Raykernel << <blocks, threads >> > (w, h, dev_bitmap, (World *)dev_world);

	cudaMemcpy(host_bitmap, dev_bitmap, imageSize, cudaMemcpyDeviceToHost);
}


__global__ void worldKernel(const int w, const int h, World* dev_world) {
	dev_world->bg_color = RGBColor(0, 0, 0);

	dev_world->vp.hres = w;
	dev_world->vp.vres = h;
	dev_world->vp.pixel_size = 1;
	dev_world->vp.gamma = 1.0;
	dev_world->vp.inv_gamma = 1.0;
	dev_world->vp.set_sampler(new Sample_MultiJittered());

	dev_world->camera_ptr = new Camera_Pinhole(Vec3(0, 50, 300), Vec3(0, 0, 0));

	Sphere *sphere_ptr = new Sphere(Vec3(0, 50, 0), 50);
	sphere_ptr->color = RGBColor(1, 0, 0);
	dev_world->add_object(sphere_ptr);

	//	Sphere *sphere_ptr = new Sphere(Vec3(0, -20, 0), 60);
	//	sphere_ptr->color = RGBColor(0, 1, 0);
	//	dev_world->add_object(sphere_ptr);

	Plane *plane_ptr = new Plane(Vec3(0, 0, 0), Vec3(0, 1, 0));
	plane_ptr->color = RGBColor(0.8, 0.8, 0.8);
	dev_world->add_object(plane_ptr);
}


World* InitWorld(const int w, const int h) {
	World *dev_world = nullptr;
	cudaMalloc(&dev_world, sizeof(World));
	worldKernel << <1, 1 >> > (w, h, dev_world);

	printf("123");
	return dev_world;
}