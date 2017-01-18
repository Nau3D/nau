#version 430

//uniform sampler2DArray shadowMaps; 
uniform sampler2DShadow shadowMaps; 

in vec3 texCoordV;
in vec4 viewSpacePos;
in vec4 modelPos;
in vec4 projShadowCoord;
in vec3 normalV, lightDir;

layout (location = 0) out vec4 outPos;
//layout (location = 1) out vec4 outNormal;
layout (location = 2) out vec4 outColor;

void main()
{
	outPos = vec4(modelPos);
	//outNormal = vec4(normalize(normalV)*0.5+0.5, 1.0);
	
	vec4 color = vec4(0.0,0.0,0.0,1.0);
	float w = projShadowCoord.w;
	vec4 p = projShadowCoord/ w;
		
	vec3 n = normalize(normalV);

	float intensity = max (dot (n, lightDir), 0.0);
	float a=0,b=0;
	float dispX = 0.5/textureSize(shadowMaps,0).x; //displacement for X
	float dispY = 0.5/textureSize(shadowMaps,0).y; //displacement for Y
	
  	if (intensity > 0.0) { 
	
		vec2 p0 = vec2(p.xy + vec2(-dispX, -dispX));
		vec2 p1 = vec2(p.xy + vec2(dispX, dispX));
		vec2 p2 = vec2(p.xy + vec2(dispX, -dispX));
		vec2 p3 = vec2(p.xy + vec2(-dispX, dispX));
		
		float s0 = texture(shadowMaps, vec3(p0.xy, p.z)); //Top Left
		float s1 = texture(shadowMaps, vec3(p1.xy, p.z));  //BottomRight
		float s2 = texture(shadowMaps, vec3(p2.xy, p.z));  //Top Right
		float s3 = texture(shadowMaps, vec3(p3.xy, p.z));  //Bottom Left
		//float s4 = texture(shadowMaps, vec3(vec2(p.xy + vec2(0, 0)), p.z)); //Center
		
		float d = (s0+s1+s2+s3) / 4.0f;		
		
		bool light = (d >= 1.0f); //os quatro dissem que esta em luz
		bool shadow = (d <= 0.0f); // os quatro dissem que esta em sombra
		
		if (light) { //os quatro dizem que esta em Luz
			color = vec4(0.0,0.0,1.0,1.0);
		}
		else {
			if (shadow) { //os quatro dizem que esta em Sombra
				color = vec4(0.0,0.0,0.0,1.0);
			}
			else { //os quatro discordem entre si (Duvida)
				color = vec4(1.0,0.0,0.0,1.0);
			}
		}
	} 
	else {//Como Non Light Facing (PVCL)
		color = vec4(0.0,1.0,0.0,1.0); 
	}
	outColor = color;	//vec4(0,0,b,0);
}

