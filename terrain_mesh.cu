#include "terrain_mesh.cuh"

__global__ void make_mesh(Vec3f* positions, Vec3f* colors, Vec3f* normals, Vec2f* textCoords) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int threadI = threadIdx.x;
	int blockI = blockIdx.x;

	//tri 1
	//vert 1
	positions[(i * 6) + 0].x = threadI;
	positions[(i * 6) + 0].y = 0;
	positions[(i * 6) + 0].z = blockI;

	colors[(i * 6) + 0].x = 0.5;
	colors[(i * 6) + 0].y = 0.5;
	colors[(i * 6) + 0].z = 0.5;

	textCoords[(i * 6) + 0].x = 0;
	textCoords[(i * 6) + 0].y = 0;


	//vert 2
	positions[(i * 6) + 1].x = threadI + 1;
	positions[(i * 6) + 1].y = 0;
	positions[(i * 6) + 1].z = blockI + 1;

	colors[(i * 6) + 1].x = 0.5;
	colors[(i * 6) + 1].y = 0.5;
	colors[(i * 6) + 1].z = 0.5;

	textCoords[(i * 6) + 1].x = 1;
	textCoords[(i * 6) + 1].y = 1;


	//vert 3
	positions[(i * 6) + 2].x = threadI + 1;
	positions[(i * 6) + 2].y = 0;
	positions[(i * 6) + 2].z = blockI;

	colors[(i * 6) + 2].x = 0.5;
	colors[(i * 6) + 2].y = 0.5;
	colors[(i * 6) + 2].z = 0.5;

	textCoords[(i * 6) + 2].x = 0;
	textCoords[(i * 6) + 2].y = 1;


	//tri 2
	//vert 4
	positions[(i * 6) + 3].x = threadI + 1;
	positions[(i * 6) + 3].y = 0;
	positions[(i * 6) + 3].z = blockI + 1;

	colors[(i * 6) + 3].x = 0.5;
	colors[(i * 6) + 3].y = 0.5;
	colors[(i * 6) + 3].z = 0.5;

	textCoords[(i * 6) + 3].x = 1;
	textCoords[(i * 6) + 3].y = 1;


	//vert 5
	positions[(i * 6) + 4].x = threadI;
	positions[(i * 6) + 4].y = 0;
	positions[(i * 6) + 4].z = blockI;

	colors[(i * 6) + 4].x = 0.5;
	colors[(i * 6) + 4].y = 0.5;
	colors[(i * 6) + 4].z = 0.5;

	textCoords[(i * 6) + 4].x = 0;
	textCoords[(i * 6) + 4].y = 0;


	//vert 6
	positions[(i * 6) + 5].x = threadI;
	positions[(i * 6) + 5].y = 0;
	positions[(i * 6) + 5].z = blockI + 1;

	colors[(i * 6) + 5].x = 0.5;
	colors[(i * 6) + 5].y = 0.5;
	colors[(i * 6) + 5].z = 0.5;

	textCoords[(i * 6) + 5].x = 1;
	textCoords[(i * 6) + 5].y = 0;

	//normals
	normals[(i * 6) + 0].x = ((positions[(i * 6) + 1].z - positions[(i * 6) + 0].z) * (positions[(i * 6) + 2].y - positions[(i * 6) + 0].y)) - ((positions[(i * 6) + 1].y - positions[(i * 6) + 0].y) * (positions[(i * 6) + 2].z - positions[(i * 6) + 0].z));
	normals[(i * 6) + 0].y = ((positions[(i * 6) + 1].x - positions[(i * 6) + 0].x) * (positions[(i * 6) + 2].z - positions[(i * 6) + 0].z)) - ((positions[(i * 6) + 1].z - positions[(i * 6) + 0].z) * (positions[(i * 6) + 2].x - positions[(i * 6) + 0].x));
	normals[(i * 6) + 0].z = ((positions[(i * 6) + 1].y - positions[(i * 6) + 0].y) * (positions[(i * 6) + 2].x - positions[(i * 6) + 0].x)) - ((positions[(i * 6) + 1].x - positions[(i * 6) + 0].x) * (positions[(i * 6) + 2].y - positions[(i * 6) + 0].y));

	normals[(i * 6) + 1].x = normals[(i * 6) + 0].x;
	normals[(i * 6) + 1].y = normals[(i * 6) + 0].y;
	normals[(i * 6) + 1].z = normals[(i * 6) + 0].z;

	normals[(i * 6) + 2].x = normals[(i * 6) + 0].x;
	normals[(i * 6) + 2].y = normals[(i * 6) + 0].y;
	normals[(i * 6) + 2].z = normals[(i * 6) + 0].z;


	normals[(i * 6) + 3].x = ((positions[(i * 6) + 4].z - positions[(i * 6) + 3].z) * (positions[(i * 6) + 5].y - positions[(i * 6) + 3].y)) - ((positions[(i * 6) + 4].y - positions[(i * 6) + 3].y) * (positions[(i * 6) + 5].z - positions[(i * 6) + 3].z));
	normals[(i * 6) + 3].y = ((positions[(i * 6) + 4].x - positions[(i * 6) + 3].x) * (positions[(i * 6) + 5].z - positions[(i * 6) + 3].z)) - ((positions[(i * 6) + 4].z - positions[(i * 6) + 3].z) * (positions[(i * 6) + 5].x - positions[(i * 6) + 3].x));
	normals[(i * 6) + 3].z = ((positions[(i * 6) + 4].y - positions[(i * 6) + 3].y) * (positions[(i * 6) + 5].x - positions[(i * 6) + 3].x)) - ((positions[(i * 6) + 4].x - positions[(i * 6) + 3].x) * (positions[(i * 6) + 5].y - positions[(i * 6) + 3].y));

	normals[(i * 6) + 4].x = normals[(i * 6) + 3].x;
	normals[(i * 6) + 4].y = normals[(i * 6) + 3].y;
	normals[(i * 6) + 4].z = normals[(i * 6) + 3].z;

	normals[(i * 6) + 5].x = normals[(i * 6) + 3].x;
	normals[(i * 6) + 5].y = normals[(i * 6) + 3].y;
	normals[(i * 6) + 5].z = normals[(i * 6) + 3].z;
}

__device__ inline float atomicCAS(float* addr, float compare, float val)
{
	int old = *addr;
	*addr = (old == compare) ? val : old;
	return old;
}

__device__ inline float atomicMin(float* addr, float value) {
	float old = *addr;
	float assumed = min(old, value);
	while(atomicCAS(addr, old, assumed) != old)
	{
		old = *addr;
		assumed = min(old, value);
	}
}

__global__ void find_tri_intersect(Vec3f* ray_wor, Vec3f* positions, Vec3f* normals, float* dist_from_O, Camera* cam, Vec3f* insect_p) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int threadI = threadIdx.x;
	int blockI = blockIdx.x;
	int* b;

	Vec3f E1 = positions[(i * 3) + 1] - positions[(i * 3) + 0];
	Vec3f E2 = positions[(i * 3) + 2] - positions[(i * 3) + 0];
	Vec3f N = cross(E1, E2);
	float det = -dot(*ray_wor, N);
	float invdet = 1.0 / det;
	Vec3f AO = -cam->pos - positions[(i * 3) + 0];
	Vec3f DAO = cross(AO, *ray_wor);
	float u = dot(E2, DAO) * invdet;
	float v = -dot(E1, DAO) * invdet;
	float t = dot(AO, N) * invdet;
	if(det<=-1e-6 && t<=0.0 && u>=0.0 && v>=0.0 && (u+v) <= 1.0)
	{
		Vec3f temp_insect = -cam->pos + (*ray_wor * t);

		float temp_magnitude = magnitude(cam->pos - temp_insect);
		atomicMin(dist_from_O, temp_magnitude);
		if(*dist_from_O == temp_magnitude)
		{
			insect_p->x = temp_insect.x;
			insect_p->y = -temp_insect.y;
			insect_p->z = temp_insect.z;
		}
	}
}

__global__ void resize_terrain(Vec3f* positions, Vec3f* colors, Vec3f* normals, float peak_radius, float decline_radius, float peak_height, Vec3f* p) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int threadI = threadIdx.x;
	int blockI = blockIdx.x;

	for (int j = 0; j < 6; j++) {
		if (pow(positions[(i * 6) + j].x - p->x, 2) + pow(positions[(i * 6) + j].z - p->z, 2) < pow(peak_radius, 2)) {
			positions[(i * 6) + j].y += peak_height;
			if (positions[(i * 6) + j].y < 0)
				positions[(i * 6) + j].y = 0;
		}
		else if (pow(positions[(i * 6) + j].x - p->x, 2) + pow(positions[(i * 6) + j].z - p->z, 2) < pow(decline_radius + peak_radius - 1, 2)) {
			float m = -(peak_height / decline_radius);
			float x = magnitude(*p - positions[(i * 6) + j]);
			x = abs(x - peak_radius);
			float c = peak_height;
			float incline = (m * x) + c;

			if (p->y <= positions[(i * 6) + j].y && incline > 0) {
				positions[(i * 6) + j].y += incline;
			}
		}
	}

	//normals
	normals[(i * 6) + 0].x = ((positions[(i * 6) + 1].z - positions[(i * 6) + 0].z) * (positions[(i * 6) + 2].y - positions[(i * 6) + 0].y)) - ((positions[(i * 6) + 1].y - positions[(i * 6) + 0].y) * (positions[(i * 6) + 2].z - positions[(i * 6) + 0].z));
	normals[(i * 6) + 0].y = ((positions[(i * 6) + 1].x - positions[(i * 6) + 0].x) * (positions[(i * 6) + 2].z - positions[(i * 6) + 0].z)) - ((positions[(i * 6) + 1].z - positions[(i * 6) + 0].z) * (positions[(i * 6) + 2].x - positions[(i * 6) + 0].x));
	normals[(i * 6) + 0].z = ((positions[(i * 6) + 1].y - positions[(i * 6) + 0].y) * (positions[(i * 6) + 2].x - positions[(i * 6) + 0].x)) - ((positions[(i * 6) + 1].x - positions[(i * 6) + 0].x) * (positions[(i * 6) + 2].y - positions[(i * 6) + 0].y));

	normals[(i * 6) + 1].x = normals[(i * 6) + 0].x;
	normals[(i * 6) + 1].y = normals[(i * 6) + 0].y;
	normals[(i * 6) + 1].z = normals[(i * 6) + 0].z;

	normals[(i * 6) + 2].x = normals[(i * 6) + 0].x;
	normals[(i * 6) + 2].y = normals[(i * 6) + 0].y;
	normals[(i * 6) + 2].z = normals[(i * 6) + 0].z;


	normals[(i * 6) + 3].x = ((positions[(i * 6) + 4].z - positions[(i * 6) + 3].z) * (positions[(i * 6) + 5].y - positions[(i * 6) + 3].y)) - ((positions[(i * 6) + 4].y - positions[(i * 6) + 3].y) * (positions[(i * 6) + 5].z - positions[(i * 6) + 3].z));
	normals[(i * 6) + 3].y = ((positions[(i * 6) + 4].x - positions[(i * 6) + 3].x) * (positions[(i * 6) + 5].z - positions[(i * 6) + 3].z)) - ((positions[(i * 6) + 4].z - positions[(i * 6) + 3].z) * (positions[(i * 6) + 5].x - positions[(i * 6) + 3].x));
	normals[(i * 6) + 3].z = ((positions[(i * 6) + 4].y - positions[(i * 6) + 3].y) * (positions[(i * 6) + 5].x - positions[(i * 6) + 3].x)) - ((positions[(i * 6) + 4].x - positions[(i * 6) + 3].x) * (positions[(i * 6) + 5].y - positions[(i * 6) + 3].y));

	normals[(i * 6) + 4].x = normals[(i * 6) + 3].x;
	normals[(i * 6) + 4].y = normals[(i * 6) + 3].y;
	normals[(i * 6) + 4].z = normals[(i * 6) + 3].z;

	normals[(i * 6) + 5].x = normals[(i * 6) + 3].x;
	normals[(i * 6) + 5].y = normals[(i * 6) + 3].y;
	normals[(i * 6) + 5].z = normals[(i * 6) + 3].z;
}

__global__ void highlight_terrain(Vec3f* positions, Vec3f* colors, float peak_radius, float decline_radius, float peak_height, Vec3f* p) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int threadI = threadIdx.x;
	int blockI = blockIdx.x;

	for (int j = 0; j < 6; j++) {
		if (pow(positions[(i * 6) + j].x - p->x, 2) + pow(positions[(i * 6) + j].z - p->z, 2) < pow(peak_radius, 2)) {
			colors[(i * 6) + j].x = 0;
			colors[(i * 6) + j].y = 0;
			colors[(i * 6) + j].z = 1;
		}
		else if (pow(positions[(i * 6) + j].x - p->x, 2) + pow(positions[(i * 6) + j].z - p->z, 2) < pow(decline_radius + peak_radius - 1, 2)) {
			colors[(i * 6) + j].x = 0.2;
			colors[(i * 6) + j].y = 0.2;
			colors[(i * 6) + j].z = 0.5;
		}
		else
		{
			colors[(i * 6) + j].x = 0.5;
			colors[(i * 6) + j].y = 0.5;
			colors[(i * 6) + j].z = 0.5;
		}
		/*else if (positions[(i * 6) + j].y == 0) {
			colors[(i * 6) + j].x = 0.5;
			colors[(i * 6) + j].y = 0.5;
			colors[(i * 6) + j].z = 0.5;
		}
		else if (positions[(i * 6) + j].y > 0) {
			colors[(i * 6) + j].x = positions[(i * 6) + j].y / 10;
			colors[(i * 6) + j].y = 0;
			colors[(i * 6) + j].z = 0;
		}
		else if (positions[(i * 6) + j].y < 0) {
			colors[(i * 6) + j].x = 0;
			colors[(i * 6) + j].y = abs(positions[(i * 6) + j].y) / 10;
			colors[(i * 6) + j].z = 0;
		}*/
	}
}