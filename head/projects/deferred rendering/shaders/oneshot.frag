#version 430

uniform int texCount;
uniform sampler2D texUnit;
uniform vec4 diffuse;

in vec3 Normal;
in vec2 TexCoord;

uniform vec3 lightDirection;

uniform mat4 V;

in vec2 texCoordV;
out vec4 colorOut;

void main() {
	
	
	vec3 n = normalize(Normal);
	vec3 ld = vec3(V * vec4(-lightDirection, 0.0));
	float intensity = max(0.0, dot(n, normalize(ld)));
	
	vec4 color;
	if (texCount != 0)
		color = texture(texUnit, TexCoord);
	else
		color = diffuse;
	colorOut = vec4(color.rgb*intensity+0.2, color.a);
	//colorOut = vec4(n,1);
}
