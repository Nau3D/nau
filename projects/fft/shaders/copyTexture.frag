#version 430

uniform sampler2D texUnit0;

in vec2 texCoord;

out vec4 outc;

void main () {
	vec2 comp;
	
	comp = texture(texUnit0, texCoord).rg;
	// compute the magnitude
	float mag = sqrt(comp.x*comp.x + comp.y*comp.y);
	// apply log2 for display purposes
	float k = log2(mag+1e-20);
	outc = vec4(k,k,k, 0);
}