#version 430

in vec4 position;
in vec3 normal;
in vec4 texCoord0;

out vec4 texPos;
out vec3 normalVar;
out vec4 posVar;

uniform mat4 PVM;
uniform mat4 M;
uniform mat4 VM;
uniform mat3 Normal;

void main(void) {
	texPos = texCoord0;
	normalVar = normalize(Normal*normal);
	normalVar = normalize(vec3(M*vec4(normal,0.0)));
	posVar = M * position;
	gl_Position = PVM * position;
}
