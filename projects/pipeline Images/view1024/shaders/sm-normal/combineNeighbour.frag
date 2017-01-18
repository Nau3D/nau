#version 430

uniform sampler2DShadow shadowMaps; 
//uniform sampler2DArrayShadow shadowMaps;

#define EPSILON 0.00001

in vec3 texCoordV;
in vec4 viewSpacePos;
in vec4 projShadowCoord;
in vec3 normalV, lightDir;

out vec4 outColor;

void texelCoherence() {
	
	vec4 color = vec4(0.0,0.0,0.0,1.0);
	float w = projShadowCoord.w;
	vec4 p = projShadowCoord/ w;
		
	vec3 n = normalize(normalV);

	float intensity = max (dot (n, lightDir), 0.0);
	float disp = 1.0/textureSize(shadowMaps,0).x;
	
 	if (intensity > 0.0) {
		float d0 = texture(shadowMaps, vec3(vec2(p.xy + vec2(0, 0)), p.z)); 			//Texel 0,0 
		float d1 = texture(shadowMaps, vec3(vec2(p.xy + vec2(-disp, -disp)), p.z));		//Texel -1,-1
		float d2 = texture(shadowMaps, vec3(vec2(p.xy + vec2(-disp, 0)), p.z)); 		//Texel -1,0 
		float d3 = texture(shadowMaps, vec3(vec2(p.xy + vec2(0, -disp)), p.z)); 		//Texel 0,-1		
		
		float coherence = (d0+d1+d2+d3) / 4.0; 
		
		if (coherence == 1.0) { //Está em luz
			color = vec4(0.0,0.0,1.0,1.0);
		}
		else { //esta em sombra
			if (coherence == 0.0){
				color = vec4(0.0,0.0,0.0,1.0);
			}
			else {
				color = vec4(1.0,0.0,0.0,1.0);
			}
		}
		
	} 
	else {//Conto como se estivesse mesmo em sombra
		color = vec4(0.0,0.0,0.0,1.0);
	}
	outColor = color;	//vec4(0,0,b,0);

}

void normal() {
	vec4 color = vec4(0.0,0.0,0.0,1.0);
	float w = projShadowCoord.w;
	vec4 p = projShadowCoord/ w;
		
	vec3 n = normalize(normalV);

	float intensity = max (dot (n, lightDir), 0.0);
	float disp = 1.0/textureSize(shadowMaps,0).x;
	
 	if (intensity > 0.0) {
		//Pixels Virados para Luz (PVPL)
		float d0 = texture(shadowMaps, vec3(vec2(p.xy), p.z)); 			//Texel 0,0 
		
		bool emluz = (d0>= 1.0);
		bool sombra = (d0== 0.0);
		
		if (emluz) { //Está em luz
			color = vec4(0.0,0.0,1.0,1.0);
		}
		else { //esta em sombra
			if (sombra){
				color = vec4(0.0,0.0,0.0,1.0);
			}
/* 			else {//Problema?
				color = vec4(1.0,0.0,0.0,1.0);
			} */
		}
	} 
	else {//Pixels Virados Contra a Luz (PVCL)
		color = vec4(0.0,1.0,0.0,1.0);
	}
	outColor = color;	//vec4(0,0,b,0);

}

void main()
{
	normal();
	//neighbouringTexels1();
	//texelCoherence();
}
