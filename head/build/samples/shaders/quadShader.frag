#version 330

uniform sampler2D texUnit;

in vec2 texCoordV;

out vec4 outColor;

void main() {


	outColor = texture(texUnit, texCoordV);

}