#version 440

uniform vec4 diffuse;
uniform float shininess;
uniform int texCount;
uniform sampler2D texUnit;


in vec3 lightDirV;
in vec3 normalV;
in vec2 texCoordV;
in vec4 worldPosV;

layout (location = 0) out vec4 colorOut;
layout (location = 1) out vec4 posRSM;

void main() {

	float intensity = max(dot(normalize(normalV),lightDirV), 0.0);

	if (texCount == 0) {
		colorOut = (intensity) * diffuse;
	}
	else {
		colorOut = ((intensity) * diffuse) * texture(texUnit, texCoordV);
	}
	posRSM = worldPosV;
}