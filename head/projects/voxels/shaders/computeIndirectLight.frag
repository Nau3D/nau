#version 440

in vec2 texCoordV;

out vec4 outColor;

uniform sampler2D texPos;
uniform sampler2D texNormal;
uniform sampler2D texColor;
uniform sampler3D grid;
uniform int GridSize;


vec4 voxelConeTrace(vec3 origin, vec3 dir, float coneRatio, float maxDist) {

	vec3 samplePos = origin;
	vec4 accum = vec4(0.0);
	
	float minDiameter = 2.0/GridSize;
	
	float startDist = 1 * minDiameter;
	float dist = startDist;
	
	while(dist < maxDist && accum.a < 1.0) {
	
		float sampleDiameter = max(minDiameter, coneRatio * dist);
		
		// convert diameter to LOD
		// for example:
		// -log2(1/256) = 8
		// -log2(1/128) = 7
		// -log2(1/64) = 6
		float sampleLOD = -log2(sampleDiameter);
		
		vec3 samplePos = origin + dir * (dist);
		float level =  max(0, log2(GridSize) - sampleLOD  );
		
		//if (sampleLOD >= 5 && sampleLOD < 6) 
		{
		vec4 sampleValue = textureLod(grid, samplePos , level);
		float sampleWeight = (1.0 - accum.w);
		//accum.a += sampleValue.a * sampleWeight;
		accum.rgb += sampleValue.rgb * sampleWeight;//* sampleValue.a ;
//		accum.rgb += sampleValue.rgb ;//* sampleValue.a;
		sampleValue.a = 1.0 - pow(1.0 - sampleValue.a, minDiameter/sampleDiameter);
		//sampleValue.a /= minDiameter/sampleDiameter;
		accum.a += sampleValue.a;// * minDiameter/sampleDiameter ;
		// if (level > 2.0)
			// accum = 0.5;
			
		// else accum = 0.0;	
		}
		dist += sampleDiameter/2.0;
		//dist += 1000;
	}
	return accum;
}

void main()
{
	vec4 color = texture(texColor, texCoordV);
	vec3 coord = texture(texPos, texCoordV).xyz;
	ivec3 coordi = ivec3(coord * GridSize);
	//coord = coordi* 1.0/GridSize;
	vec4 texel = texelFetch(grid, coordi, 0);
//	if (texel.w == 0)
//		discard;
//	vec3 normal = texelFetch(gridNormal, coordi, 0).xyz;
	vec3 normal = texture(texNormal, texCoordV).xyz ;
	normal = normalize(normal);
	vec3 tangent, bitangent;
	vec3 c1 = cross(normal, vec3(0,0,1));
	vec3 c2 = cross(normal, vec3(0,1,0));
	if (length(c1) > length(c2)) 
		tangent = c1;
	else 
		tangent = c2;
		
	tangent = normalize(tangent);
	bitangent = cross(normal, tangent);
	
	float coneRatio = 0.677;
	float maxDist = 1.0;
	vec4 il=vec4(0);
	
	float sbeta = sin(45);
	float cbeta = cos(45);
	float alpha = 60;
	float alpha2 = 30;
	il  = voxelConeTrace(coord, normalize(normal), coneRatio, maxDist);
	  il  += 0.5 * voxelConeTrace(coord, normalize(normal*sbeta+tangent*cbeta), coneRatio, maxDist);
	  il  += 0.5 * voxelConeTrace(coord, normalize(normal*sbeta-tangent*cbeta), coneRatio, maxDist);
	  il  += 0.5 * voxelConeTrace(coord, normalize(normal*sbeta+tangent*0.5*cbeta + bitangent*0.866*cbeta), coneRatio, maxDist);
	  il  += 0.5 * voxelConeTrace(coord, normalize(normal*sbeta+tangent*0.5*cbeta - bitangent*0.866*cbeta), coneRatio, maxDist);
	  il  += 0.5 * voxelConeTrace(coord, normalize(normal*sbeta-tangent*0.5*cbeta + bitangent*0.866*cbeta), coneRatio, maxDist);
	  il  += 0.5 * voxelConeTrace(coord, normalize(normal*sbeta-tangent*0.5*cbeta - bitangent*0.866*cbeta), coneRatio, maxDist);
	// ao += 0.707 * voxelConeTrace(coord, normalize(normal+tangent), coneRatio, maxDist);
	// ao += 0.707 * voxelConeTrace(coord, normalize(normal-tangent), coneRatio, maxDist);z
	// ao += 0.707 * voxelConeTrace(coord, normalize(normal+bitangent), coneRatio, maxDist);
	// ao += 0.707 * voxelConeTrace(coord, normalize(normal-bitangent), coneRatio, maxDist);
	
	outColor = vec4((color *il*0.05)) ;
	//outColor = vec4(il*0.0125);
	//outColor = vec4(normal*0.5 + 0.5,0);
	texel = texelFetch(grid, coordi/4, 2);
	//float a = 1.0 - pow(1.0 - texel.a,255);
	//outColor = vec4(texel);
	//outColor =  color;//vec4(normal,1);
}