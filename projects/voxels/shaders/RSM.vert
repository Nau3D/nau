#version 440

in vec4 position;
in vec3 normal;
in vec2 texCoord0;

uniform mat4 PVM;
uniform vec3 lightDir;

out vec3 lightDirV;
out vec3 normalV;
out vec2 texCoordV;
out vec4 worldPosV;

void main() {

	lightDirV = - normalize(lightDir);
	normalV = normalize(normal);
	texCoordV = texCoord0;
	worldPosV = position;
	gl_Position = PVM * position;
}