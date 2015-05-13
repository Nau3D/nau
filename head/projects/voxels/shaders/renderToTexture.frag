#version 440

in vec3 normalV;
in vec2 texCoordV;
in vec4 posV;

out vec4 outColor;

uniform float shininess;
uniform vec4 diffuse;
uniform int texCount;
uniform sampler2D texUnit;

void main()
{
	outColor = posV*0.5 + 0.5;
}