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
		
		float distance = -viewSpacePos.z /  viewSpacePos.w;

        if (distance < split1)
            color += diffuse * (NdotL * textureProj(shadowMap1, projShadowCoord[0])) * vec4(1.0, 0.0, 0.0, 1.0);
        else if (distance < split2)
            color += diffuse * (NdotL * textureProj(shadowMap2, projShadowCoord[1])) * vec4(0.0, 1.0, 0.0, 1.0);
        else if (distance < split3)
            color += diffuse * (NdotL * textureProj(shadowMap3, projShadowCoord[2])) * vec4(1.0, 0.0, 1.0, 1.0);
        else if (distance < split4)
            color += diffuse * (NdotL * textureProj(shadowMap4, projShadowCoord[3])) * vec4(0.0, 0.0, 1.0, 1.0);

	}
	outColor = color;	
}
