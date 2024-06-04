#ifndef CAMERA_H
#define CAMERA_H

#include <math.h>
#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <vector>
#include "device_launch_parameters.h"
#include "imgui_internal.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "vec2.cuh"
#include "mat44.cuh"

struct Camera {
	Vec3f pos;
	Vec3f front;
	Vec3f up;
	Vec3f cameraDir;
	Vec3f camUp;
	Vec3f camRight;
	float cameraRight;
	float cameraUp;
	bool cameraState = false;
	bool cameraClick = false;
	float lastX;
	float lastY;
	float xoffset;
	float yoffset;
};

inline
Camera findDir(Camera cam) {
	cam.cameraDir.x = cam.pos.x - cam.front.x;
	cam.cameraDir.y = cam.pos.y - cam.front.y;
	cam.cameraDir.z = cam.pos.z - cam.front.z;
	cam.cameraDir.x = 1;
	return cam;
}

inline
Vec3f normalize(Vec3f v) {
	float length_of_v = sqrt((v.x * v.x) + (v.y * v.y) + (v.z * v.z));
	return Vec3f{ v.x / length_of_v, v.y / length_of_v, v.z / length_of_v };
}

#endif
