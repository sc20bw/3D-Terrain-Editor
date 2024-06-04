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
#pragma comment(lib, "glew32.lib")
#include <gl/glew.h>
#include <gl/GL.h>
#include <gl/GLU.h>
#include <GLFW/glfw3.h>
#include <vector>
#include "terrain_mesh.cuh"

arrayMesh create_terrain();