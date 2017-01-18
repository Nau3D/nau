#version 330

uniform vec4 lightDirection, lightColor;
uniform vec4 diffuse, ambient, emission;
uniform float shininess;
uniform int texCount;
uniform sampler2D texUnit;

in vec3 Normal;
in vec2 TexCoord;
in vec3 LightDirection;
in vec4 VertexPos;

layout (location = 0) out vec4 outColor;
layout (location = 1) out vec4 outPositions;

void main()
{
	vec4 color;
	vec4 amb;
	float intensity;
	vec4 lightIntensityDiffuse;
	vec3 lightDir;
	vec3 n;
	
	lightDir = -normalize(LightDirection);
	n = normalize(Normal);	
	intensity = max(dot(lightDir,n),0.0);
	
	lightIntensityDiffuse = lightColor * intensity;
	
	if (texCount == 0) {
		color = diffuse * lightIntensityDiffuse + diffuse * 0.3 ;
	}
	else {
		color = (diffuse * lightIntensityDiffuse + emission + 0.3) * texture(texUnit, TexCoord);
	}
	outColor = vec4(vec3(color ), color.a);
//	outColor = vec4(color.r, 1.0, 1.0, 1.0);
	outPositions = vec4(vec3(VertexPos),intensity);
}
