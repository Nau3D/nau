#version 430

uniform sampler2D texUnit;

in vec2 texCoord;

out vec4 real;

void main () {
	
	real = vec4(texture(texUnit, texCoord).r);
}