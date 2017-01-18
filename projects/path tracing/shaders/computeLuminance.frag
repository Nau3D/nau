#version 430

in vec2 texPos;

uniform sampler2D input;

out vec2 luminance;

void main() {
	
	vec4 c2 = texture(input, texPos);
	float lumi = (0.2126*c2.r)+(0.0722*c2.g)+(0.7152*c2.b);
	luminance = vec2(lumi*16);
}	