#version 440

in vec4 position;
in vec2 texCoord0;

uniform mat4 PVM;

out vec2 texCoordV;

void main()
{
	texCoordV = texCoord0;
    gl_Position = PVM * position;
}
