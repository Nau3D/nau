#version 430

in vec4 position;
in vec4 texCoord0;

out vec4 texCoord;

void main()
{
	texCoord = texCoord0;
	gl_Position = position;
}