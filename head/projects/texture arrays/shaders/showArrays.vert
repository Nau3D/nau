#version 330


in vec4 position;
in vec4 texCoord0;


out vec2 texCoordV;

void main() {

	gl_Position = position;
	texCoordV = vec2(texCoord0);	
}