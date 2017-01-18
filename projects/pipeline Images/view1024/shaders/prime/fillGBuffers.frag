#version 430

uniform int texCount;
uniform sampler2D texUnit;
uniform vec4 diffuse;

uniform int RenderTargetX;
uniform int RenderTargetY;

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
	vec3 n = normalize(Normal);
	float NdotL = max(0.0,dot (n, LightDirection));
	
	outPos = vec4(Pos);
	outNormal = vec4(n, 0.0);
	/* outNormal = vec4(normalize(Normal)*0.5+0.5, 0.0); */
	vec4 auxColor;
	//auxColor = vec4(NdotL,0.0,0.0,1.0);
	
	if (NdotL > 0.0 ){
		auxColor = vec4(1.0);
/* 		if (texCount != 0)
			auxColor = texture(texUnit, TexCoord);
		else
			auxColor = diffuse; */
	}
	else {
		auxColor = vec4(0.0,0.0,0.0,1.0);
	}	
	
	
	outColor = vec4(auxColor);
}
