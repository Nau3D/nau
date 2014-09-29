#version 330

uniform mat4 PVM,VM,M;

in vec4 position;
in vec4 texCoord0;

out vec4 texPos;

void main(void) {
	texPos = texCoord0;

	gl_Position = PVM * position;
}
