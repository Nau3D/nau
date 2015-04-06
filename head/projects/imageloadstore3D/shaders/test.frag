#version 440

uniform usampler3D texUnit;

in vec2 texCoordV;

out vec4 outColor;

void main() {

	uint r = texelFetch(texUnit, ivec3(0,0,0),0).r;
	if (r == 852)
		outColor = vec4(1,0,0,1);
	else
		outColor = vec4(0,1,0,1);

}