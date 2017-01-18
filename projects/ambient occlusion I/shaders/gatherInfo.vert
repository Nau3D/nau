#version 330

in vec4 position;
in vec3 normal;

out vec4 normalVar;
out vec4 posVar;

uniform mat4 PVM;
uniform mat4 VM;
uniform mat3 Normal;

void main(void) {
	normalVar = vec4(normalize(Normal*normal), 0.0);
	posVar = VM * position;
	gl_Position = PVM * position;
}
