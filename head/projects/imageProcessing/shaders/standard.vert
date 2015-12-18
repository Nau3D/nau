#version 440

in vec4 position;
in vec2 texCoord0;

out vec2 texCoordV;

void main() {

	gl_Position = position;
	texCoordV = texCoord0;
} 
