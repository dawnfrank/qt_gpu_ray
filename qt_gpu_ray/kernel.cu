
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "scenedata.h"
#include "define.h"

#include "sphere.cuh"
#include "plane.cuh"
#include "world.cuh"
#include "camera_pinhole.cuh"
#include "sample_multijitered.cuh"

extern "C" World* InitWorld(const int w, const int h);
extern "C" void ModifyWorld(SceneData sData, World* dev_world);
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
		Vec2 sp;
		Vec3 pp;

		double pixel_size = dev_world->vp.pixel_size;
		int sample_num = dev_world->vp.get_samples_num();
		//		double disZ = (dev_world->camera_ptr->eye - dev_world->camera_ptr->at).magnitude();
		Mat3 inverseCameraMatrix = dev_world->camera_ptr->coord.inverse();
		ray.origin = dev_world->camera_ptr->eye;

		//		for (int s = 0; s != sample_num; ++s) {
		sp = dev_world->vp.sampler_ptr->sample_unit_square();
		pp.x = pixel_size*(x - 0.5*(w - 1) + sp.x);
		pp.y = -pixel_size*(y - 0.5*(h - 1) + sp.y);
		pp.z = 1000;

//		if (x == 400 && y == 500)printf("%f %f %f %f %f %f %\n", pp*dev_world->camera_ptr->coord, dev_world->camera_ptr->eye);

		//pp = pp*inverseCameraMatrix;
		pp = pp*dev_world->camera_ptr->coord;

		ray.direction = Vec3(pp.x, pp.y, -pp.z);

		sr = dev_world->hit_objects(ray);
		pixel_color += sr.color;
		//		}
		//		pixel_color /= sample_num;

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
	dev_world->vp.hres = w;
	dev_world->vp.vres = h;
	dev_world->vp.pixel_size = 1;
	dev_world->vp.gamma = 1.0;
	dev_world->vp.inv_gamma = 1.0;
	dev_world->vp.set_sampler(new Sample_MultiJittered());

	dev_world->camera_ptr = new Camera_Pinhole(Vec3(0, 100, 1000), Vec3(0, 50, 0));

	Sphere *sphere_ptr = new Sphere(Vec3(0, 50, 0), 50);
	sphere_ptr->color = RGBColor(1, 0, 0);
	dev_world->add_object(sphere_ptr);

	Plane *plane_ptr = new Plane(Vec3(0, 0, 0), Vec3(0, 1, 0));
	plane_ptr->color = RGBColor(0.8, 0.8, 0.8);
	dev_world->add_object(plane_ptr);
}


World* InitWorld(const int w, const int h) {
	World *dev_world = nullptr;
	cudaMalloc(&dev_world, sizeof(World));
	worldKernel << <1, 1 >> > (w, h, dev_world);

	return dev_world;
}

__global__ void modifyKernel(int cameraCmd, int cameraRotateX, int cameraRotateY, int cameraZoom, World* dev_world) {
	Vec3 moveDir;
	if (cameraCmd & CAMERA_CMD_W) moveDir.z -= CAMERA_MOVE_DIS;
	if (cameraCmd & CAMERA_CMD_S) moveDir.z += CAMERA_MOVE_DIS;
	if (cameraCmd & CAMERA_CMD_A) moveDir.x -= CAMERA_MOVE_DIS;
	if (cameraCmd & CAMERA_CMD_D) moveDir.x += CAMERA_MOVE_DIS;
	if (moveDir.x != 0 || moveDir.y != 0 || moveDir.z != 0) dev_world->camera_ptr->move(moveDir);


	printf("%d %d\n", cameraRotateX, cameraRotateY);
	if (cameraRotateX != 0 || cameraRotateY != 0) dev_world->camera_ptr->rotate(Vec3(cameraRotateX, cameraRotateY, 0));

	if (cameraZoom != 0)dev_world->camera_ptr->zoom(cameraZoom);

}

void ModifyWorld(SceneData sData, World* dev_world)
{
	int cameraCmd = sData.cameraCmd;
	int cameraRotateX = sData.cameraRotateX;
	int cameraRotateY = sData.cameraRotateY;
	int cameraZoom = sData.cameraZoom;
	modifyKernel << <1, 1 >> > (cameraCmd, cameraRotateX, cameraRotateY, cameraZoom, dev_world);
}
