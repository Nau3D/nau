#version 420


in vec4 position;
in vec4 texCoord0;

out Data {
	vec4 texCoord;
} DataOut;

void main()
{
	DataOut.texCoord = texCoord0;

	gl_Position = position ;
} 
