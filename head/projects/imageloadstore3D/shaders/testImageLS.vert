#version 440

uniform mat4 PVM;
uniform mat3 NormalMatrix;

in vec4 position;
in vec4 normal;
in vec4 texCoord0;

out vec3 Normal;
out vec4 Position;

void main()
{
	Position = position;
	Normal = normalize(NormalMatrix * vec3(normal));
	gl_Position = PVM * position;
}