#version 440

in vec4 position;
in vec3 normal;
in vec2 texCoord0;

uniform mat4 PVM;
uniform mat3 NormalMat;
uniform mat4 ViewModelMat;
uniform vec3 lightDir;

out vec3 normalV;
out vec2 texCoordV;
out vec4 posV;
out vec3 lightDirV;

void main()
{
	lightDirV = - normalize(lightDir);
	texCoordV = texCoord0;
    gl_Position = PVM * position;
	normalV = normalize(normal);
	posV = position;
}
