#version 330

in vec4 position;
in vec4 normal;
in vec4 texCoord0;

out vec2 TexCoordv;
out vec3 Normalv;

void main()
{
	Normalv = vec3(normal);
	TexCoordv = vec2(texCoord0);
	gl_Position = position;
}