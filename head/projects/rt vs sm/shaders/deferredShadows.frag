#version 330

uniform sampler2D shadowMap, shadowMapBF, contourProj;
uniform vec4 lightDirection, lightColor;
uniform vec4 diffuse, ambient, emission;
uniform float shininess;
uniform int texCount;
uniform sampler2D texUnit;

in vec3 Normal;
in vec2 TexCoord;
in vec3 LightDirection;
in vec4 VertexPos;
in vec4 projShadowCoord;

layout (location = 0) out vec4 outColor;
layout (location = 1) out vec4 shadows;
layout (location = 2) out vec4 shadowContourProj;
layout (location = 3) out vec4 outPositions;

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
	intensity = max(dot(lightDir,n),0.0f);
	
	lightIntensityDiffuse = lightColor * intensity;
	
	if (texCount < 5) {
		color = diffuse * lightIntensityDiffuse + diffuse * 0.3f ;
	}
	else {
		color = (diffuse * lightIntensityDiffuse + emission + 0.3f) * texture(texUnit, TexCoord);
	}
	
	float d = texture(shadowMapBF, projShadowCoord.xy).r;
	vec4 s;
	
	if (d < projShadowCoord.z)
		s.x = 0.0f;
	else
		s.x = 1.0f;

	if (intensity == 0.0f)
		s.y = 0.0f;
	else
		s.y = 0.01f;

	shadows = s;
		
	float f = texture(contourProj, projShadowCoord.xy).r;

	float res;
	if (f == 0 )
		res = 0.0f;
	else
		res = 1.0f;
		
	shadowContourProj = vec4(res);
		
	outColor = vec4(vec3(color ), color.a);
//	outColor = vec4(color.r, 1.0, 1.0, 1.0);
	outPositions = vec4(vec3(VertexPos),res);
}
