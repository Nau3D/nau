#version 430


in vec4 position;
in vec4 texCoord0;

out vec3 texCoordV;

void main() {

	
	gl_Position = position;
	texCoordV = vec3(texCoord0);	
}