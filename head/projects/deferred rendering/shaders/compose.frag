#version 430

uniform sampler2D texColor;
uniform sampler2D texNormal;
uniform vec3 lightDirection;

uniform mat4 V;

in vec2 texCoordV;
out vec4 colorOut;

void main() {
	
	vec4 color = texture(texColor, texCoordV);
	vec3 n = texture(texNormal, texCoordV).xyz * 2.0 - 1.0;
	vec3 ld = vec3(V * vec4(-lightDirection, 0.0));

	float intensity = max(0.0, dot(n, normalize(ld)));
	colorOut = vec4(color.rgb*intensity+0.2, color.a);
}
