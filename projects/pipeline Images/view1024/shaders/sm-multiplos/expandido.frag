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
	float disp = 0.5/textureSize(shadowMaps,0).x; //para imagens com tamanhos diferentes, criar um disp para os y's
	
 	if (intensity > 0.0) {
		//float d0 = texture(shadowMaps, vec4(p.xy,0,p.z)); //Shadow Map encolhido
		float s0 = texture(shadowMaps, vec4(vec2(p.xy + vec2(-disp, -disp)), 1, p.z)); //Top Left
		float s1 = texture(shadowMaps, vec4(vec2(p.xy + vec2(disp, disp)), 1, p.z) );  //BottomvRight
		float s2 = texture(shadowMaps, vec4(vec2(p.xy + vec2(disp, -disp)), 1, p.z));  //Top Right
		float s3 = texture(shadowMaps, vec4(vec2(p.xy + vec2(-disp, disp)), 1, p.z));  //Bottom Left
		float s4 = texture(shadowMaps, vec4(vec2(p.xy + vec2(0, 0)), 1, p.z)); //Center
		
		float d1 = min( s0, min( s1, min( s2, min( s3, s4 ) ) ) ); //isto não é bem no centro do pixel.
		//float d1 = texture(shadowMaps, vec4(p.xy,1,p.z)); //Shadow Map expandido	
		
		if (d1 >= 1 ) { //está em Luz
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