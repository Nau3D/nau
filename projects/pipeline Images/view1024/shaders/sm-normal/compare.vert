#version 430

in vec4 position;
in vec4 texCoord0;

out vec2 TexCoord;

void main()
{
	TexCoord = vec2(texCoord0);
	gl_Position = position;
}