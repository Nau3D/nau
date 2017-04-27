#version 430

uniform sampler2D tex1[2], tex2[2];

in vec2 texCoordV;
out vec4 colorOut;

void main() {

	if (texCoordV.x > 0.5) {
		colorOut = texture(tex1[0], texCoordV);
	}
	else {
		colorOut = texture(tex2[1], texCoordV);
	}
}	