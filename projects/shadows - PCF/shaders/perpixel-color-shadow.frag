#version 330

uniform sampler2DShadow shadowMap;
uniform sampler2D texUnit;
uniform vec4 diffuse;
uniform int texCount;

uniform float jitter[32] = {
					0.375, 0.4375, 
					0.625, 0.0625,
					0.875, 0.1875,
					0.125, 0.0625,
					0.375, 0.6875,
					0.875, 0.4375,
					0.625, 0.5625,
					0.375, 0.9375,
					0.625, 0.3125,
					0.125, 0.5625,
					0.125, 0.8125,
					0.375, 0.1875,
					0.875, 0.9375,
					0.875, 0.6875,
					0.125, 0.3125,
					0.625, 0.8125};

uniform float jitterScale = 2;

in vec2 texCoordV;
in vec4 projShadowCoord;
in vec3 normalV, lightDir;

out vec4 outColor;

void main()
{
	vec4 color, diff;
	
	if (texCount  != 2)
		diff = texture(texUnit, texCoordV) ; 
	else 
		diff = diffuse;
		
	// ambient term
	color = diff * 0.25;
	
	vec3 n = normalize (normalV);
	
	float NdotL = max(0.0,dot (n, lightDir));
	
	float shadow = 0.0;
	if (NdotL > 0.01) {
		for (int i = 0; i < jitter.length()/2; ++i) {
			shadow += textureProj(shadowMap, projShadowCoord/projShadowCoord.w + 
				vec4(jitterScale * jitter[i*2]/4096  ,jitterScale * jitter[i*2+1]/4096 , 0.0, 0.0));
//		shadow += texture(texDepth, vec3(p + 
//				vec4(jitterScale * jitter[i*2]/512 ,jitterScale * jitter[i*2+1]/512 , 0.0, 0.0)));
		} 
		shadow /= (jitter.length()*0.5);

		color += diff  * NdotL * shadow ;
		
	}
	
	outColor = color;	
}
