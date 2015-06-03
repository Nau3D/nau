#version 440

in vec4 position;
in vec3 normal;

out vec3 normalV;

void main() {
	
	gl_Position = position;
	normalV = normal;
}

