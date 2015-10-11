#version 420

uniform	mat4 PVM;
uniform	mat4 VM;
uniform	mat3 normalMatrix;

uniform vec4 lightPos;

in vec4 position;
in vec3 normal;
in vec2 texCoord0;

out vec3 normalV;
out vec2 texCoordV;
out vec3 eyeV;
out vec3 lightDirV;

void main () {
	
	texCoordV = texCoord0;

	normalV = normalize(normalMatrix * normal);

	vec3 pos = vec3(VM * position);
	eyeV = -pos;

	vec3 lposCam = vec3(VM * lightPos);
	lightDirV = lposCam - pos;

	gl_Position = PVM * position;	
}