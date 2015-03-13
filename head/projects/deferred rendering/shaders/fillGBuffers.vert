#version 430

uniform mat4 PVM,M;
uniform mat3 NormalMatrix;


in vec4 position;
in vec4 normal;
in vec4 texCoord0;

out vec2 TexCoord;
out vec3 Normal;


void main()
{
	Normal = normalize(NormalMatrix * vec3(normal));
	TexCoord = vec2(texCoord0);
	gl_Position = PVM * position;
}