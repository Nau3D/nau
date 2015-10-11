#version 330

uniform sampler2D texUnit;

in Data {
	vec4 color;
} DataIn;

out vec4 colorOut;

void main() {

	colorOut = DataIn.color;
}