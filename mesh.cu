#include "mesh.cuh"

arrayMesh create_terrain(int terrain_width, int terrain_length) {
	int size4 = terrain_length * terrain_width * 6 * sizeof(Vec3f);
	int size2 = terrain_length * terrain_width * 6 * sizeof(Vec2f);


	Vec3f *positions = (Vec3f*)malloc(size4);
	Vec3f *colors = (Vec3f*)malloc(size4);
	Vec3f *normals = (Vec3f*)malloc(size4);
	Vec2f *textCoords = (Vec2f*)malloc(size2);

	Vec3f *device_positions = NULL;
	Vec3f *device_colors = NULL;
	Vec3f *device_normals = NULL;
	Vec2f *device_textCoords = NULL;

	cudaMalloc((void**)&device_positions, size4);
	cudaMalloc((void**)&device_colors, size4);
	cudaMalloc((void**)&device_normals, size4);
	cudaMalloc((void**)&device_textCoords, size2);

	arrayMesh TempMesh;

	make_mesh<<<terrain_length, terrain_width>>>(device_positions, device_colors, device_normals, device_textCoords);

	cudaMemcpy(positions, device_positions, size4, cudaMemcpyDeviceToHost);
	cudaMemcpy(colors, device_colors, size4, cudaMemcpyDeviceToHost);
	cudaMemcpy(normals, device_normals, size4, cudaMemcpyDeviceToHost);
	cudaMemcpy(textCoords, device_textCoords, size2, cudaMemcpyDeviceToHost);

	cudaFree(device_colors);
	cudaFree(device_positions);
	cudaFree(device_textCoords);
	cudaFree(device_normals);

	TempMesh.positions.insert(TempMesh.positions.end(), &positions[0], &positions[size4/sizeof(Vec3f)]);
	TempMesh.colors.insert(TempMesh.colors.end(), &colors[0], &colors[size4/sizeof(Vec3f)]);
	TempMesh.normals.insert(TempMesh.normals.end(), &normals[0], &normals[size4/sizeof(Vec3f)]);
	TempMesh.textCoords.insert(TempMesh.textCoords.end(), &textCoords[0], &textCoords[size2/sizeof(Vec2f)]);

	free(positions);
	free(colors);
	free(normals);
	free(textCoords);

	return TempMesh;
}

Vec3f* intersect(Vec3f* ray_wor, arrayMesh* mesh, Camera* cam, int terrain_width, int terrain_length) {
	int size4 = terrain_length * terrain_width * 6 * sizeof(Vec3f);
	int size2 = terrain_length * terrain_width * 6 * sizeof(Vec2f);
	int psize = terrain_length * terrain_width * sizeof(Vec3f);

	Vec3f* positions = (Vec3f*)malloc(size4);
	Vec3f* colors = (Vec3f*)malloc(size4);
	Vec3f* normals = (Vec3f*)malloc(size4);
	Vec2f* textCoords = (Vec2f*)malloc(size2);

	copy(mesh->positions.begin(), mesh->positions.end(), positions);
	copy(mesh->normals.begin(), mesh->normals.end(), normals);

	Vec3f* device_ray_wor;
	Vec3f* device_positions;
	Vec3f* device_normals;
	Camera* device_cam;
	Vec3f* device_p;
	float* device_dist_from_O;

	Vec3f* p = (Vec3f*)malloc(sizeof(Vec3f));
	*p = Vec3f{ 1000, 1000, 1000 };
	float* dist_from_O = (float*)malloc(sizeof(float));
	*dist_from_O = magnitude(cam->pos - (cam->pos + (*ray_wor * 1000000)));

	cudaMalloc((void**)&device_ray_wor, sizeof(Vec3f));
	cudaMalloc((void**)&device_positions, size4);
	cudaMalloc((void**)&device_normals, size4);
	cudaMalloc((void**)&device_dist_from_O, sizeof(float));
	cudaMalloc((void**)&device_cam, sizeof(Camera));
	cudaMalloc((void**)&device_p, sizeof(Vec3f));

	cudaMemcpy(device_ray_wor, ray_wor, sizeof(Vec3f), cudaMemcpyHostToDevice);
	cudaMemcpy(device_positions, positions, size4, cudaMemcpyHostToDevice);
	cudaMemcpy(device_normals, normals, size4, cudaMemcpyHostToDevice);
	cudaMemcpy(device_dist_from_O, dist_from_O, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(device_cam, cam, sizeof(Camera), cudaMemcpyHostToDevice);
	cudaMemcpy(device_p, p, sizeof(Vec3f), cudaMemcpyHostToDevice);

	find_tri_intersect<<<terrain_length, terrain_width>>>(device_ray_wor, device_positions, device_normals, device_dist_from_O, device_cam, device_p);

	cudaMemcpy(p, device_p, sizeof(Vec3f), cudaMemcpyDeviceToHost);

	cudaFree(device_ray_wor);
	cudaFree(device_positions);
	cudaFree(device_p);
	cudaFree(device_normals);
	cudaFree(device_cam);

	free(positions);
	free(colors);
	free(normals);
	free(textCoords);
	return p;
	free(p);
}

arrayMesh alter_terrain(arrayMesh *mesh, float peak_radius, float decline_radius, float peak_height, Vec3f *insect_p, int terrain_width, int terrain_length) {
	int size4 = terrain_length * terrain_width * 6 * sizeof(Vec3f);
	int size2 = terrain_length * terrain_width * 6 * sizeof(Vec2f);
	int t_w = terrain_width * 6;
	int t_l = terrain_length * 6;

	Vec3f* positions = (Vec3f*)malloc(size4);
	Vec3f* colors = (Vec3f*)malloc(size4);
	Vec3f* normals = (Vec3f*)malloc(size4);
	Vec2f* textCoords = (Vec2f*)malloc(size2);

	Vec3f* device_positions = NULL;
	Vec3f* device_colors = NULL;
	Vec3f* device_normals = NULL;
	Vec3f* device_p = NULL;

	copy(mesh->positions.begin(), mesh->positions.end(), positions);
	copy(mesh->normals.begin(), mesh->normals.end(), normals);
	copy(mesh->colors.begin(), mesh->colors.end(), colors);
	copy(mesh->textCoords.begin(), mesh->textCoords.end(), textCoords);

	cudaMalloc((void**)&device_positions, size4);
	cudaMalloc((void**)&device_colors, size4);
	cudaMalloc((void**)&device_normals, size4);
	cudaMalloc((void**)&device_p, sizeof(Vec3f));

	cudaMemcpy(device_positions, positions, size4, cudaMemcpyHostToDevice);
	cudaMemcpy(device_normals, normals, size4, cudaMemcpyHostToDevice);
	cudaMemcpy(device_colors, colors, size4, cudaMemcpyHostToDevice);
	cudaMemcpy(device_p, insect_p, sizeof(Vec3f), cudaMemcpyHostToDevice);

	arrayMesh TempMesh;

	resize_terrain <<<terrain_length, terrain_width>>> (device_positions, device_colors, device_normals, peak_radius, decline_radius, peak_height, device_p);

	cudaMemcpy(positions, device_positions, size4, cudaMemcpyDeviceToHost);
	cudaMemcpy(colors, device_colors, size4, cudaMemcpyDeviceToHost);
	cudaMemcpy(normals, device_normals, size4, cudaMemcpyDeviceToHost);

	cudaFree(device_colors);
	cudaFree(device_positions);
	cudaFree(device_p);
	cudaFree(device_normals);

	TempMesh.positions.insert(TempMesh.positions.end(), &positions[0], &positions[size4 / sizeof(Vec3f)]);
	TempMesh.colors.insert(TempMesh.colors.end(), &colors[0], &colors[size4 / sizeof(Vec3f)]);
	TempMesh.normals.insert(TempMesh.normals.end(), &normals[0], &normals[size4 / sizeof(Vec3f)]);
	TempMesh.textCoords.insert(TempMesh.textCoords.end(), &textCoords[0], &textCoords[size2 / sizeof(Vec2f)]);

	free(positions);
	free(colors);
	free(normals);
	free(textCoords);

	return TempMesh;
}

arrayMesh hover_terrain(arrayMesh* mesh, float peak_radius, float decline_radius, float peak_height, Vec3f* insect_p, int terrain_width, int terrain_length) {
	int size4 = terrain_length * terrain_width * 6 * sizeof(Vec3f);
	int size2 = terrain_length * terrain_width * 6 * sizeof(Vec2f);
	int t_w = terrain_width * 6;
	int t_l = terrain_length * 6;

	Vec3f* positions = (Vec3f*)malloc(size4);
	Vec3f* colors = (Vec3f*)malloc(size4);
	Vec3f* normals = (Vec3f*)malloc(size4);
	Vec2f* textCoords = (Vec2f*)malloc(size2);

	Vec3f* device_positions = NULL;
	Vec3f* device_colors = NULL;
	Vec3f* device_normals = NULL;
	Vec3f* device_p = NULL;

	copy(mesh->positions.begin(), mesh->positions.end(), positions);
	copy(mesh->normals.begin(), mesh->normals.end(), normals);
	copy(mesh->colors.begin(), mesh->colors.end(), colors);
	copy(mesh->textCoords.begin(), mesh->textCoords.end(), textCoords);

	cudaMalloc((void**)&device_positions, size4);
	cudaMalloc((void**)&device_colors, size4);
	cudaMalloc((void**)&device_p, sizeof(Vec3f));

	cudaMemcpy(device_positions, positions, size4, cudaMemcpyHostToDevice);
	cudaMemcpy(device_colors, colors, size4, cudaMemcpyHostToDevice);
	cudaMemcpy(device_p, insect_p, sizeof(Vec3f), cudaMemcpyHostToDevice);

	arrayMesh TempMesh;

	highlight_terrain << <terrain_length, terrain_width >> > (device_positions, device_colors, peak_radius, decline_radius, peak_height, device_p);

	cudaMemcpy(positions, device_positions, size4, cudaMemcpyDeviceToHost);
	cudaMemcpy(colors, device_colors, size4, cudaMemcpyDeviceToHost);

	cudaFree(device_colors);
	cudaFree(device_positions);
	cudaFree(device_p);

	TempMesh.positions.insert(TempMesh.positions.end(), &positions[0], &positions[size4 / sizeof(Vec3f)]);
	TempMesh.colors.insert(TempMesh.colors.end(), &colors[0], &colors[size4 / sizeof(Vec3f)]);
	TempMesh.normals.insert(TempMesh.normals.end(), &normals[0], &normals[size4 / sizeof(Vec3f)]);
	TempMesh.textCoords.insert(TempMesh.textCoords.end(), &textCoords[0], &textCoords[size2 / sizeof(Vec2f)]);

	free(positions);
	free(colors);
	free(normals);
	free(textCoords);

	return TempMesh;
}

GLuint createVAO(arrayMesh mesh) {
	unsigned int posVBO, colVBO, normVBO, texVBO, VAO = 0;
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &posVBO);
	glGenBuffers(1, &colVBO);
	glGenBuffers(1, &normVBO);
	glGenBuffers(1, &texVBO);
	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, posVBO);
	glBufferData(GL_ARRAY_BUFFER, mesh.positions.size() * 12, mesh.positions.data(), GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, colVBO);
	glBufferData(GL_ARRAY_BUFFER, mesh.colors.size() * 12, mesh.colors.data(), GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, normVBO);
	glBufferData(GL_ARRAY_BUFFER, mesh.normals.size() * 12, mesh.normals.data(), GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, texVBO);
	glBufferData(GL_ARRAY_BUFFER, mesh.textCoords.size() * 8, mesh.textCoords.data(), GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, posVBO);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, colVBO);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(1);

	glBindBuffer(GL_ARRAY_BUFFER, normVBO);
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(2);

	glBindBuffer(GL_ARRAY_BUFFER, texVBO);
	glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(3);

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glDeleteBuffers(1, &posVBO);
	glDeleteBuffers(1, &colVBO);
	glDeleteBuffers(1, &normVBO);
	glDeleteBuffers(1, &texVBO);

	return VAO;
}