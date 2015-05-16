#version 440

in vec4 position;
in vec3 normal;
in vec2 texCoord0;

uniform mat4 PVM;
uniform mat3 NormalMat;

out vec3 normalV;
out vec2 texCoordV;
out vec4 posV;

void main()
{
	texCoordV = texCoord0;
    gl_Position = PVM * position;
	normalV = normal;
	posV = position;
}
