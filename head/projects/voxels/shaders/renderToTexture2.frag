#version 440

in vec3 normalV;
in vec2 texCoordV;
in vec4 posV;

layout (location = 0) out vec4 outPos;
layout (location = 1) out vec4 outNormal;
layout (location = 2) out vec4 outColor;

uniform float shininess;
uniform vec4 diffuse;
uniform int texCount;
uniform sampler2D texUnit;
uniform mat4 LSpaceMat;
uniform sampler2D texPos;

void main()
{
	outPos = posV * 0.5 + 0.5;
	outNormal = vec4(normalV, 0);
	if (texCount != 0)
		outColor = texture(texUnit, texCoordV);
	else
		outColor = diffuse;
		
	vec4 lightTexCoord = LSpaceMat * posV;
	vec4 pos = texture(texPos, lightTexCoord.xy);
	if (distance(pos, posV) > 0.0001)
		outColor *= 0.5;
}

//ADICIONAR SOMBRAS! -> fica luz directa