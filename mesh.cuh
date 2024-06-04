#ifndef Mesh_H
#define Mesh_H

#include <stdio.h>
#include <iostream>
#include "imgui_internal.h"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#define GL_SILENCE_DEPRECATION
#define GL_GLEXT_PROTOTYPES
#ifdef _WIN32
#include <windows.h>
#endif
#include "glew-2.1.0\include\GL\glew.h"
//#include <gl/GL.h>
//#include <gl/GLU.h>
//#include <GLFW/glfw3.h>
#include <vector>
#include "terrain_mesh.cuh"

arrayMesh create_terrain(int terrain_length, int terrain_width);
Vec3f* intersect(Vec3f* ray_wor, arrayMesh* mesh, Camera* cam, int terrain_width, int terrain_length);
arrayMesh alter_terrain(arrayMesh* mesh, float peak_radius, float decline_radius, float peak_height, Vec3f* insect_p, int terrain_width, int terrain_length);
arrayMesh hover_terrain(arrayMesh* mesh, float peak_radius, float decline_radius, float peak_height, Vec3f* insect_p, int terrain_width, int terrain_length);
GLuint createVAO(arrayMesh mesh);

#endif