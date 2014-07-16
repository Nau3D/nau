#version 420

uniform mat4 PVM;
uniform mat4 M;
uniform vec3 camWorldPos;


in vec4 position;
in vec4 normal;

out vec3 normalV;
out vec3 eyeV;

void main () {
	
	normalV = normalize(vec3(M * normal));

	vec3 pos = vec3(M * position);
	
	eyeV = normalize(pos -camWorldPos);

	gl_Position = PVM * position;	
}