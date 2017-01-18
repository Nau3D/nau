#version 430

uniform sampler2D tex[2];

in vec2 texCoordV;
out vec4 colorOut;

void main() {

	if (texCoordV.x > 0.5) {
		colorOut = texture(tex[0], texCoordV);
	}
	else {
		colorOut = texture(tex[1], texCoordV);
	}
}	