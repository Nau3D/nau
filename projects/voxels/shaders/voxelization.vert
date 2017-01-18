#version 440

uniform mat4 M;

in vec4 position;
in vec3 normal;
in vec2 texCoord0;

out vec3 normalV;
out vec2 texCoordV;

void main() {
	
	gl_Position = M * position;
	normalV = normalize(normal);
	texCoordV = texCoord0;
}

