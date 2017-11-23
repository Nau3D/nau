#version 130

uniform sampler2D normal;

in vec2 texCoordV;

out vec4 outColor;

void main() {


	outColor = texture(normal, texCoordV);

}