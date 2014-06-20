#version 330

uniform sampler2DArray texUnit;

in vec2 texCoordV;

out vec4 outColor;

void main() {

	vec4 c0 = texture(texUnit, vec3(texCoordV,1));
	vec4 c1 = texture(texUnit, vec3(texCoordV,0));
	outColor = vec4(c0.r, c1.g, c1.b, 1.0);
}