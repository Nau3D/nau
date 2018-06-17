#version 440

in vec3 normalV;
in vec2 texCoordV;
in vec4 posV;
in vec3 lightDirV;

layout (location = 0) out vec4 outPos;
layout (location = 1) out vec4 outNormal;
layout (location = 2) out vec4 outColor;

uniform float shininess;
uniform vec3 diffuse;
uniform int texCount;
uniform sampler2D texUnit;
uniform mat4 LSpaceMat;
uniform sampler2D texPos;

void main()
{
	float intensity = max(0,dot(normalize(normalV), lightDirV));
	outPos = posV * 0.5 + 0.5;
	outNormal = vec4(normalV, 0);
	if (texCount != 0)
		outColor = vec4(texture(texUnit, texCoordV).xyz, intensity);
	else
		outColor = vec4(diffuse ,intensity);
		
	vec4 lightTexCoord = LSpaceMat * posV;
	vec4 pos = texture(texPos, lightTexCoord.xy);
	if (distance(vec3(pos), vec3(posV)) > 0.001)
		outColor.w = 0.0;
}

