#version 440

uniform mat4 PVM, V;

uniform mat3 NormalMatrix;

uniform vec4 lightDirection;

in vec4 normal;

in vec4 position;
layout(std430, binding = 1) buffer positions {
	vec4 pos[];
};

out vec3 Normal;
out vec3 LightDirection;

void main() {
	Normal = normalize(NormalMatrix * vec3(normal));
	LightDirection = normalize(vec3(V * lightDirection));
	gl_Position = PVM * (pos[gl_InstanceID]+position);
}

