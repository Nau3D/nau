#version 440

uniform mat4 M;

in vec4 position;
out vec4 positionV;

void main() {
	
	gl_Position = M * position;
	positionV = M * position;
}

