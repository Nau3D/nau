#version 330

uniform sampler2D texUnit;

in Data {
	flat vec4 color;
} DataIn;

out vec4 colorOut;

void main() {

	colorOut = vec4(DataIn.color);
}