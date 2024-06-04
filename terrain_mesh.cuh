#ifndef HEADER_FILE
#define HEADER_FILE
#define _USE_MATH_DEFINES

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
#include "camera.cuh"

using namespace std;

struct arrayMesh
{
	vector<Vec3f> positions;
	vector<Vec3f> colors;
	vector<Vec3f> normals;
	vector<Vec2f> textCoords;
	vector<bool> isTextured;
	vector<int> texture;
};

__global__ void make_mesh(Vec3f* positions, Vec3f* colors, Vec3f* normals, Vec2f* textCoords);

__global__ void find_tri_intersect(Vec3f* ray_wor, Vec3f* positions, Vec3f* normals, float* dist_from_O, Camera* cam, Vec3f* insect_p);

__global__ void resize_terrain(Vec3f* positions, Vec3f* colors, Vec3f* normals, float peak_radius, float decline_radius, float peak_height, Vec3f* p);

__global__ void highlight_terrain(Vec3f* positions, Vec3f* colors, float peak_radius, float decline_radius, float peak_height, Vec3f* p);

#endif