#version 430

uniform mat4 PVM;

in vec4 position;

out flat int hit;

void main(void) {
	hit = int(position.w);
	gl_Position = PVM * vec4(position.xyz,1);
}
