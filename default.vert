#version 430

layout (location = 0) in vec3 pos;
layout( location = 1 ) in vec3 iColor;
layout( location = 2 ) in vec3 iNormal;
layout( location = 3 ) in vec2 textcoords;

layout( location = 0 ) uniform mat4 model;
layout( location = 1 ) uniform mat4 view;
layout( location = 2 ) uniform mat4 proj;
layout( location = 6 ) uniform mat4 lightSpaceMatrix;

out vec3 v2fColor;
out vec3 Normal;
out vec3 FragPos;
out vec2 v2fTexCoords;
out vec4 FragPosLightSpace;

void main()
{
	FragPos = vec3(model * vec4(pos, 1.0));
	Normal = mat3(transpose(inverse(model))) * iNormal;
	v2fColor = iColor;
	v2fTexCoords = textcoords;
	FragPosLightSpace = lightSpaceMatrix * vec4(FragPos, 1.0);

	gl_Position = proj * view * vec4(pos, 1.0);
}

