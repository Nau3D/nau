#version 440

uniform mat4 PVM, V, VM,PV;

uniform mat3 NormalMatrix;

uniform vec4 lightDirection;

in vec4 normal;

in vec4 position;
layout(std430, binding = 1) buffer pswaterfall {
	vec4 pos[];
};

out vec3 Normal;
out vec3 LightDirection;
out vec3 Eye;

void main() {
	Normal = normalize(NormalMatrix * vec3(normal));
	LightDirection = vec3(V * lightDirection);
	Eye = -vec3(VM * position);
//	gl_Position = PVM * (pos[gl_InstanceID]+(position));
	vec4 p = position + pos[gl_InstanceID];
	p.w = 1.0;
	gl_Position = PV * p;
}