#version 410

uniform vec4 diffuse=vec4(1.0, 0.5, 0.5, 1.0);

uniform	mat4 viewMatrix;
uniform	sampler2D texUnit;
uniform vec3 lightDir;

in vec3 normalTE;
in vec2 texCoordTE;
out vec4 outputF;

void main() {

	vec4 color;
	vec4 amb;
	float intensity;
	vec3 n;
	
	vec3 l = normalize(vec3(viewMatrix * vec4(-lightDir,0.0)));
	n = normalize(normalTE);	
	intensity = max(dot(l,n),0.0);
	
/*	if (texCount == 0) {
		color = vec4(1.0, 0.5, 0.5, 1.0);
		amb = ambient;
	}
	else {
*/		color = texture(texUnit, texCoordTE) ;
//		amb = color * 0.33;
//	}
	outputF = (color * intensity) + color *0.2;
}