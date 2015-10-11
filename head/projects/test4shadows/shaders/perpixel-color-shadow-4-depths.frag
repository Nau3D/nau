#version 440

uniform sampler2DShadow shadowMap1,shadowMap2,shadowMap3,shadowMap4;
uniform sampler2D ;
// uniform float split[4];
uniform float split1,split2,split3, split4;

in vec4 viewSpacePos;

in vec4 projShadowCoord[4];
in vec3 normalV, lightDir;

out vec4 outColor;

void main()
{
	vec4 color = vec4(0.4);
	vec4 diffuse = vec4(1.0);
		
	vec3 n = normalize (normalV);

	float NdotL = max (dot (n, lightDir), 0.0);
	
	if (NdotL > 0.0) {
//		color += diffuse * NdotL;
	
		
		 float split[4];
		 split[0] = split1;
		 split[1] = split2;
		 split[2] = split3;
		 split[3] = split4;	
	
		float distance = -viewSpacePos.z /  viewSpacePos.w;
		
		//sfloat distance = -viewSpacePos.z;
		float f1= textureProj(shadowMap1, projShadowCoord[0]);
		vec4 p = projShadowCoord[1];
		//p.z *= p.w;
		float f2= textureProj(shadowMap2, projShadowCoord[1]);
		float f3= textureProj(shadowMap3, projShadowCoord[2]);
		float f4= textureProj(shadowMap4, projShadowCoord[3]);
		// vec4 f1= vec4(textureProj(shadowMap[0], projShadowCoord[0]));
		// vec4 f2= vec4(textureProj(shadowMap[1], projShadowCoord[1]));
		// vec4 f3= vec4(textureProj(shadowMap[2], projShadowCoord[2]));
		// vec4 f4= vec4(textureProj(shadowMap[3], projShadowCoord[3]));
		for (int i = 0; i < 4; i++) {
			if (distance < split[i]) {
 				if (i == 0) {
					color += diffuse * (NdotL * textureProj(shadowMap1, projShadowCoord[0])) * vec4(1.0, 0.0, 0.0, 1.0);
					//color += diffuse * NdotL * textureGather (shadowMap[i], projShadowCoord[i].xy, projShadowCoord[i].z/projShadowCoord[i].w) * vec4(1.0, 0.0, 0.0, 1.0);
				}
				else if (i == 1){
					color += diffuse * (NdotL * textureProj(shadowMap2, projShadowCoord[1])) * vec4(0.0, 1.0, 0.0, 1.0);
				}
				else if (i == 2) {
					color += diffuse * NdotL * f3 * vec4(1.0, 0.0, 1.0, 1.0);
				}
				else if (i == 3) {
					color += diffuse * NdotL * f4 * vec4(0.0, 0.0, 1.0, 1.0);
				}
 			//float depth = projShadowCoord[i].z / projShadowCoord[i].w;
				//float depthShadow = shadow2DProj (shadowMap[i], projShadowCoord[i]).r;
				//if (depthShadow > depthShadow) {
					//color *= 0.0;
				
				/*	if (0 == i) {
						color = f1;// vec4 (1.0, 0.0, 0.0, 1.0);
					}				
					else if (1 == i) {
						color = f2;//vec4 (0.0, 1.0, 0.0, 1.0);
					}				
					else if (2 == i) {
						color = f3;//vec4 (0.0, 0.0, 1.0, 1.0);
					}				
					else if (3 == i) {
						color = f4;//vec4 (0.0, 1.0, 1.0, 1.0);
					}*/
								
			//	}
				break;		

			}
		}
	}
	outColor = color;	
}
