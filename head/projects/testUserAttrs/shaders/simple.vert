#version 330

uniform mat4 PVM, V, VM;
uniform mat3 NormalMat;
uniform vec4 lightDir;

in vec3 normal;
in vec4 position;
in vec4 texCoord0;

out vec3 normalV;
out vec2 texCoordV;
out vec3 lightDirV;
out float depth;

void main() {
	
	lightDirV = normalize(vec3(V * (-lightDir)));
	normalV = normalize(NormalMat * normal);
	texCoordV = vec2(texCoord0);
	depth = -(VM * position).z;
	gl_Position = PVM * position;
}