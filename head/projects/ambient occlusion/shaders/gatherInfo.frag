#version 430 

#define SAMPLES 9

in vec3 normalVar;
in vec4 posVar;
in vec4 texPos;

layout (location = 0) out vec4 normalMap;
layout (location = 1) out vec4 posMap;

void main(void) {
		
	vec3 normal = normalize(normalVar);
	normalMap = vec4((vec3(normal))* 0.5 + 0.5, 1.0);
	posMap = posVar;
}
