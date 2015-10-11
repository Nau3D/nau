#version 430

uniform int texCount;
uniform sampler2D texUnit;
uniform vec4 diffuse;

in vec4 Pos;
in vec3 Normal;
in vec2 TexCoord;
in vec3 LightDirection;
in vec3 LightDirW;

layout (location = 0) out vec4 outPos;
layout (location = 1) out vec4 outNormal;
layout (location = 2) out vec4 outColor;

void main()
{
	outPos = vec4(Pos);
	outNormal = vec4(normalize(Normal)*0.5+0.5, 0.0);
	vec4 auxColor;
	if (texCount != 0)
		auxColor = texture(texUnit, TexCoord);
	else
		auxColor = diffuse;
	outColor = vec4(auxColor);
}
