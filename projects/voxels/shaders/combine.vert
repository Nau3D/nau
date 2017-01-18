#version 430

in vec4 position;
in vec2 texCoord0;

out vec2 texPos;

void main(void) {

	texPos = texCoord0;
	gl_Position = position;
}
