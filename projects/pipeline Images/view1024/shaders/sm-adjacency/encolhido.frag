#version 430

//uniform sampler2DArray shadowMaps; 
uniform sampler2DArrayShadow shadowMaps;

in vec3 texCoordV;
in vec4 viewSpacePos;
in vec4 projShadowCoord;
in vec3 normalV, lightDir;

out vec4 outColor;

void main()
{
	vec4 color = vec4(0.0,0.0,0.0,1.0);
	float w = projShadowCoord.w;
	vec4 p = projShadowCoord/ w;
		
	vec3 n = normalize(normalV);

	float intensity = max (dot (n, lightDir), 0.0);
	float a=0,b=0;
	
 	if (intensity > 0.0) {
		float d0 = texture(shadowMaps, vec4(p.xy,0,p.z)); //Shadow Map encolhido
		//float d1 = texture(shadowMaps, vec4(p.xy,1,p.z)); //Shadow Map expandido	
		
		if (d0 >= 1 ) { //está em Luz
				color = vec4(0.0,0.0,1.0,1.0);
		}
 		else { //Está em Sombra
			color = vec4(0.0,0.0,0.0,1.0);
		}
	} 
	else {//Conto como se estivesse mesmo em sombra
		color = vec4(0.0,1.0,0.0,1.0);
	}
	outColor = color;	//vec4(0,0,b,0);
}
