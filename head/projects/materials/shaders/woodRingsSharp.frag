#version 330

uniform vec4 lightColor = vec4(0.44, 0.21, 0.0, 1.0);
uniform vec4 darkColor = vec4(0.91, 0.5, 0.12, 1.0);
uniform float frequency = 32;
uniform float sharpness = 16;

in vec4 worldPos;

out vec4 colorOut;

void main() {

	float blend = sin(length(worldPos.xz) * frequency) * 0.5 + 0.5 ;
	blend = pow(blend, 1/sharpness);
	vec4 dif = mix(lightColor, darkColor, blend);
	colorOut = dif;
}