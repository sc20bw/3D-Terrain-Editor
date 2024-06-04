#include <stdio.h>
#include <iostream>
#include <cmath>
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
#include "glfw-master\include\GLFW\glfw3.h"
#include "stb_image.h"
#include "mesh.cuh"
#include "shader.cuh"

using namespace std;

Camera cam;

int display_w, display_h;

float t;
bool cameraFast = false;
float cameraSpeed = 0.05;

Vec3f p;
Vec3f ray_wor;
Vec3f ray;
Vec3f ray_nds;
Vec4f ray_clip;
Mat44f proj;
Mat44f view;
Mat44f model;

bool space_press = false;

GLuint load_texture_2d(char const* aPath)
{
    assert(aPath);
    stbi_set_flip_vertically_on_load(true);
    int w, h, channels;
    stbi_uc* ptr = stbi_load(aPath, &w, &h, &channels, 4);
    if (!ptr)
        cout << "can't load image";
    // Generate texture object and initialize texture with image
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_SRGB8_ALPHA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, ptr);
    stbi_image_free(ptr);
    // Generate mipmap hierarchy
    glGenerateMipmap(GL_TEXTURE_2D);
    // Configure texture
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
    return tex;
}

void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    cam.xoffset = xpos - cam.lastX;
    cam.yoffset = cam.lastY - ypos; // reversed since y-coordinates go from bottom to top

    cam.lastX = xpos;
    cam.lastY = ypos;
}

void processInput(GLFWwindow* window) {
    if (cam.cameraState == true) {
        cam.cameraRight += cam.xoffset * 0.05;
        cam.cameraUp += cam.yoffset * 0.05;
        cam.xoffset = 0;
        cam.yoffset = 0;

        Vec3f direction;
        direction.x = cos(cam.cameraRight * M_PI / 180) * cos(cam.cameraUp * M_PI / 180);
        direction.y = sin(cam.cameraUp * M_PI / 180);
        direction.z = sin(cam.cameraRight * M_PI / 180) * cos(cam.cameraUp * M_PI / 180);
        cam.front = normalize(direction);

        if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        {
            cameraFast = true;
        }
        if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_RELEASE) {
            cameraFast = false;
        }
        if(cameraFast == true)
        {
            cameraSpeed = 0.5;
        }
        else
        {
            cameraSpeed = 0.05;
        }
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
            cam.pos -= cameraSpeed * cam.front;
            //cam.pos.x = cam.pos.x + (0.1 * sin(cam.cameraRight));
        }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
            cam.pos += cameraSpeed * cam.front;
            //cam.pos.x = cam.pos.x - (0.1 * sin(cam.cameraRight));
        }
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
            cam.pos += cameraSpeed * normalize(cross(cam.front, cam.camUp));
            //cam.pos.z = cam.pos.z + (0.1 * sin(cam.cameraRight));
        }
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
            cam.pos -= cameraSpeed * normalize(cross(cam.front, cam.camUp));
            //cam.pos.z = cam.pos.z - (0.1 * sin(cam.cameraRight));
        }
    }

    if (cam.cameraClick == false) {
        if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
            cout << ray_wor.x << ", " << ray_wor.y << ", " << ray_wor.z << "\n";
            cam.cameraClick = true;
        }
    }

    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS && space_press == false) {    
        space_press = true;
        if (cam.cameraState == false) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
            cam.cameraState = true;
            cam.lastX = display_w / 2;
            cam.lastY = display_h / 2;
        }
        else {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            cam.cameraState = false;
        }
    }

    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_RELEASE) {
        space_press = false;
    }
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_RELEASE) {
        cam.cameraClick = false;
    }
}

int main(int argc, const char* argv[])
{
    int terrain_width = 100;
    int terrain_length = 100;

    /*Mat44f m = Mat44f{
        5,9,1,3,
        4,4,8,8,
        2,10,8,6,
        1,0,3,11
    };

    cout << determinent(m);

    Mat44f invM = inverse(m);

    for (int i = 0; i < 16; i++) {
        if (i % 4 == 0) {
            cout << "\n";
        }
        cout << m.v[i] << ", ";
    }

    cout << "\n";

    for (int i = 0; i < 16; i++) {
        if (i % 4 == 0) {
            cout << "\n";
        }
        cout << invM.v[i] << ", ";
    }

    cout << "\n";*/

    if (!glfwInit())
        return 1;

    // Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
    // GL ES 2.0 + GLSL 100
    const char* glsl_version = "#version 100";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(__APPLE__)
    // GL 3.2 + GLSL 150
    const char* glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
    // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_TRUE);
    glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    //glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only

    //glfwWindowHint(GLFW_DEPTH_BITS, 24);
#endif

    // Create window with graphics context
    GLFWwindow* window = glfwCreateWindow(1920, 1080, "3D Terrain Editor", nullptr, nullptr);
    if (window == nullptr)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSwapInterval(1); // Enable vsync

    glewInit();

    glEnable(GL_FRAMEBUFFER_SRGB);
    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);

    Shader shader("default.vert", "default.frag");
    shader.use();

    arrayMesh mesh = create_terrain(terrain_width, terrain_length);
    GLuint ourVAO = createVAO(mesh);

    /*for (int i = 0; i < 6 * terrain_length * terrain_width; i++) {
        cout << mesh.positions.at(i).x << ", " << mesh.positions.at(i).y << ", " << mesh.positions.at(i).z << "\n";
    }*/

    /*for (int i = 0; i < 6 * terrain_length * terrain_width; i++) {
        cout << mesh.colors.at(i).x << ", " << mesh.colors.at(i).y << ", " << mesh.colors.at(i).z << ", " << i << "\n";
    }*/

    unsigned int depthMapFBO;
    glGenFramebuffers(1, &depthMapFBO);

    const unsigned int SHADOW_WIDTH = 1024, SHADOW_HEIGHT = 1024;
    unsigned int depthMap;
    glGenTextures(1, &depthMap);
    glBindTexture(GL_TEXTURE_2D, depthMap);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT,
        SHADOW_WIDTH, SHADOW_HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);

    glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMap, 0);
    glDrawBuffer(GL_NONE);
    glReadBuffer(GL_NONE);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    cam.pos = Vec3f{ 0, -1, 0 };
    cam.front = Vec3f{ 0, 0, 1 };
    cam.up = Vec3f{ 0, -1, 0 };
    cam.cameraRight = 0;
    cam.cameraUp = 0;
    proj = make_perspective_projection(45 * (M_PI / 180), 800 / 600, 0.1, 100);

    Vec3f* ray_intersect = (Vec3f*)malloc(sizeof(Vec3f));

    float peak_height = 0.1f;
    float peak_radius = 0.5f;
    float decline_radius = 0.001f;
    GLuint texture = load_texture_2d("./textures/mountain.jfif");
    bool polygon_mode = true;

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsLight();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Our state
    bool show_demo_window = false;
    bool show_another_window = false;
    ImVec4 clear_color = ImVec4(0.1f, 0.1f, 0.1f, 1.00f);

    float i = 0;

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

    // Main loop
#ifdef __EMSCRIPTEN__
    // For an Emscripten build we are disabling file-system access, so let's not attempt to do a fopen() of the imgui.ini file.
    // You may manually call LoadIniSettingsFromMemory() to load settings from your own storage.
    io.IniFilename = nullptr;
    EMSCRIPTEN_MAINLOOP_BEGIN
#else
    while (!glfwWindowShouldClose(window))
#endif
    {
        processInput(window);
        // Poll and handle events (inputs, window resize, etc.)
        // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
        // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application, or clear/overwrite your copy of the mouse data.
        // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application, or clear/overwrite your copy of the keyboard data.
        // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();


        static int counter = 0;

        // 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
        if (show_demo_window)
            ImGui::ShowDemoWindow(&show_demo_window);


        ImGui::Begin("#CH", nullptr, ImGuiWindowFlags_NoMove | ImGuiInputTextFlags_ReadOnly | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoScrollbar);
        auto draw = ImGui::GetBackgroundDrawList();
        draw->AddCircle(ImVec2(display_w / 2, display_h / 2), 6, IM_COL32(255, 0, 0, 255), 100, 0.0f);
        ImGui::End();

        // 2. Show a simple window that we create ourselves. We use a Begin/End pair to create a named window.
        ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.

        ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
        ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
        ImGui::Checkbox("Another Window", &show_another_window);

        ImGui::SliderFloat("Peak radius:", &peak_radius, 0.001f, 10.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
        ImGui::SliderFloat("peak height:", &peak_height, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
        ImGui::SliderFloat("decline radius:", &decline_radius, 0.001f, 10.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
        ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

        if (ImGui::Button("Polygon Mode")) {                            // Buttons return true when clicked (most widgets return true when edited/activated)
            polygon_mode = !polygon_mode;
        }
        ImGui::SameLine();
        ImGui::Text("counter = %d", counter);

        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
        ImGui::End();

        // 3. Show another simple window.
        if (show_another_window)
        {
            ImGui::Begin("Another Window", &show_another_window);   // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
            ImGui::Text("Hello from another window!");
            if (ImGui::Button("Close Me"))
                show_another_window = false;
            ImGui::End();
        }

        // Rendering
        ImGui::Render();
        glfwGetFramebufferSize(window, &display_w, &display_h);

        Mat44f lightProj, lightView, lightSpaceMatrix;
        float near_plane = 1.0f, far_plane = 7.5f;
        Vec3f lightDir, lightRight, lightUp, lightPos, lightTarget, lightAbove;
        lightProj = make_ortho(-10.f, 10.f, -10.f, 10.0f, near_plane, far_plane);
        lightPos = Vec3f{ 0.0f, 6.0f, 0.0f };
        lightTarget = Vec3f{ 0.0f, 0.0f, 0.0f };
        lightAbove = Vec3f{ 0.0f, 0.1f, 0.0f };
        lightDir = normalize(lightPos - (lightPos + lightAbove));
        lightRight = normalize(cross(lightDir, lightAbove));
        lightUp = normalize(cross(lightDir, lightRight));
        lightView = make_direction_matrix(lightRight, lightUp, lightDir);
        lightSpaceMatrix = lightProj * lightView;

        glBindFramebuffer(GL_FRAMEBUFFER, depthMapFBO);
        glClear(GL_DEPTH_BUFFER_BIT);
        glActiveTexture(GL_TEXTURE0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (polygon_mode) {
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        }
        else {
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        }

        shader.use();

        cam.cameraDir = normalize(cam.pos - (cam.pos + cam.front));
        cam.camRight = normalize(cross(cam.cameraDir, cam.up));
        cam.camUp = normalize(cross(cam.cameraDir, cam.camRight));

        model = kIdentity44f; //make_rotation_x(180 * M_PI / 180) /** make_rotation_y(-45 * M_PI / 180)*/;
        view = make_direction_matrix(cam.camRight, cam.camUp, cam.cameraDir);
        Mat44f viewTrans = (view * make_translation(cam.pos));

        ray_wor = -(1/magnitude(cam.front))*cam.front;

        /*Vec3f E1 = mesh.positions[(0 * 3) + 1] - mesh.positions[(0 * 3) + 0];
        Vec3f E2 = mesh.positions[(0 * 3) + 2] - mesh.positions[(0 * 3) + 0];
        Vec3f N = cross(E1, E2);
        float det = -dot(ray_wor, N);
        float invdet = 1.0 / det;
        Vec3f AO = -cam.pos - mesh.positions[(0 * 3) + 0];
        Vec3f DAO = cross(AO, ray_wor);
        float u = dot(E2, DAO) * invdet;
        float v = -dot(E1, DAO) * invdet;
        float t = dot(AO, N) * invdet;

        cout << "det = " << det << " t = " << t << " u = " << u << " v = " << v << "\n";*/

        ray_intersect = intersect(&ray_wor, &mesh, &cam, terrain_width*2, terrain_length*2);
        if (cam.cameraClick == true) {
            cout << ray_intersect->x << ", " << ray_intersect->y << ", " << ray_intersect->z << "\n";
            mesh = alter_terrain(&mesh, peak_radius, decline_radius, peak_height, ray_intersect, terrain_width, terrain_length);
        }
        mesh = hover_terrain(&mesh, peak_radius, decline_radius, peak_height, ray_intersect, terrain_width, terrain_length);
        ourVAO = createVAO(mesh);

        glUniformMatrix4fv(0, 1, GL_TRUE, model.v);
        glUniformMatrix4fv(1, 1, GL_TRUE, viewTrans.v);
        glUniformMatrix4fv(2, 1, GL_TRUE, proj.v);

        glUniform3f(3, 0.0f, 6.0f, 0.0f);
        glUniform3f(4, cam.pos.x, cam.pos.y, cam.pos.z);
        glUniform3f(5, 1.0f, 1.0f, 1.0f);
        glUniformMatrix4fv(6, 1, GL_TRUE, lightSpaceMatrix.v);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, depthMap);

        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, texture);

        glBindVertexArray(ourVAO);
        glDrawArrays(GL_TRIANGLES, 0, terrain_length * terrain_width * 6 * 50);
        glBindVertexArray(0);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        i = i + 1;

        glfwSwapBuffers(window);
    }
#ifdef __EMSCRIPTEN__
    EMSCRIPTEN_MAINLOOP_END;
#endif

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}