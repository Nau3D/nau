#version 330 

in vec4 normalVar;
in vec4 posVar;

layout (location = 0) out vec4 normalMap;
layout (location = 1) out vec4 posMap;

void main(void) {
	//	normalMap = vec4((normalize(vec3(normalVar))* 0.5) + 0.5, 1.0);
		normalMap = vec4(normalize(vec3(normalVar)), 1.0);
		posMap = posVar;
}
