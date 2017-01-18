#version 330

in Data {
	vec4 color;
} DataIn;

out vec4 outputF;

void main() {

	outputF = DataIn.color;
}