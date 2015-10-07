#version 410

layout (std140) uniform Material {
	vec4 diffuse;
	vec4 ambient;
	vec4 specular;
	vec4 emissive;
	float shininess;
	int texCount;
};

uniform	sampler2D texUnit;
uniform vec3 lightDir;

in Data{
	vec3 normal;
	vec2 texCoord;
};

out vec4 outputF;

void main()
{
	vec4 color, emission, amb;
	float intensity;
	vec3 l, n;
	
	l = normalize(lightDir);
	n = normalize(normal);	
	intensity = max(dot(l,n),0.0);
	
	if (texCount == 0) {
		color = diffuse;
		amb = ambient;
		emission = emissive;
	}
	else {
		color = texture(texUnit, texCoord);// * diffuse;
		amb = texture(texUnit, texCoord) * vec4(0.8);
		emission = texture(texUnit, texCoord) * emissive;
	}
	outputF = (color  /*intensity*/);// + amb + emission;
}
