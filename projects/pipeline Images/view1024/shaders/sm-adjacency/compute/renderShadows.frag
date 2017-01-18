#version 430

//uniform sampler2DArray shadowMaps; 
uniform sampler2DArrayShadow shadowMaps;

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
		//Shadow Map Encolhido
		float c0 = texture(shadowMaps, vec4(vec2(p.xy + vec2(-dispX, -dispX)), 0, p.z)); //Top Left
		float c1 = texture(shadowMaps, vec4(vec2(p.xy + vec2(dispX, dispX)), 0, p.z) );  //BottomvRight
		float c2 = texture(shadowMaps, vec4(vec2(p.xy + vec2(dispX, -dispX)), 0, p.z));  //Top Right
		float c3 = texture(shadowMaps, vec4(vec2(p.xy + vec2(-dispX, dispX)), 0, p.z));  //Bottom Left
		float c4 = texture(shadowMaps, vec4(vec2(p.xy + vec2(0, 0)), 1, p.z)); //Center
		
		//float d0 = texture(shadowMaps, vec4(p.xy,0,p.z)); //Shadow Map encolhido
		float d0 = max( c0, max( c1, max( c2, max( c3, c4 ) ) ) ); //isto não é bem no centro do pixel.
		
		//Shadow Map expandido
		float s0 = texture(shadowMaps, vec4(vec2(p.xy + vec2(-dispX, -dispX)), 1, p.z)); //Top Left
		float s1 = texture(shadowMaps, vec4(vec2(p.xy + vec2(dispX, dispX)), 1, p.z) );  //BottomvRight
		float s2 = texture(shadowMaps, vec4(vec2(p.xy + vec2(dispX, -dispX)), 1, p.z));  //Top Right
		float s3 = texture(shadowMaps, vec4(vec2(p.xy + vec2(-dispX, dispX)), 1, p.z));  //Bottom Left
		float s4 = texture(shadowMaps, vec4(vec2(p.xy + vec2(0, 0)), 1, p.z)); //Center
		
		float d1 = min( s0, min( s1, min( s2, min( s3, s4 ) ) ) ); //isto não é bem no centro do pixel.
		
		if (d1 >= 1 ) {
			if ( d0 >=1 ) { //esta em luz no expandido e no encolhido
				//color = vec4(1.0,1.0,1.0,1.0);
				color = vec4(0.0,0.0,1.0,1.0);
			}
			else {//esta em luz no expandido mas em sombra no encolhido (como??)
				color = vec4(1.0,1.0,0.0,1.0); //duvida (S/L)
			}
		}
 		else { 
			if (d0 < 1 ) { //esta em sombra no expandido e no encolhido && intensity > 0.4 
					color = vec4(0.0,0.0,0.0,1.0);
			}
			else { //esta em sombra no expandido mas em luz no encolhido
				color = vec4(1.0,0.0,0.0,1.0); //duvida (L/S)
			}
		}
	} 
	else {//Como Non Light Facing (PVCL)
		color = vec4(0.0,1.0,0.0,1.0); 
	}
	outColor = color;	//vec4(0,0,b,0);
}

