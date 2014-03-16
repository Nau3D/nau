#version 330

uniform vec4 diffuse;
uniform vec4 ambient;
uniform vec4 lightDir;
uniform vec4 emission;

in vec3 normalV;
in vec2 texCoordV;
in vec3 lightDirV;

out vec4 outColor;

void main() {

	vec3 l = normalize(vec3(-lightDirV));
	vec3 n = normalize(vec3(normalV));
	float i = max(dot(l,n),0.0);

	outColor = diffuse * i + ambient + emission;

}