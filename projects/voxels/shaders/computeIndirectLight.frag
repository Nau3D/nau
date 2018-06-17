#version 440

in vec2 texCoordV;

out vec4 outColor;

uniform sampler2D texPos;
uniform sampler2D texNormal;
uniform sampler2D texColor;
uniform sampler3D grid, gridN;
uniform int GridSize;
uniform vec3 camPos;
uniform int mode=0;


vec4 voxelConeTrace(vec3 origin, vec3 dir, float coneRatio, float maxDist) {

	vec3 samplePos = origin;
	vec4 accum = vec4(0.0);
	float ao = 0;
	float minDiameter = 2.0/GridSize;
	
	float startDist = 2 * minDiameter;
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
		
		vec4 sampleValue = textureLod(grid, samplePos , level);
		sampleValue.a /= minDiameter/sampleDiameter;
		accum.rgb += sampleValue.rgb  * (1-accum.a);		
		accum.a += sampleValue.a;
		dist += sampleDiameter*2;
	}
	accum.a = ao;
	return accum;
}


void main()
{
	vec4 color = texture(texColor, texCoordV);
	vec3 coord = texture(texPos, texCoordV).xyz;
	ivec3 coordi = ivec3(coord * GridSize);
	vec4 texel = texelFetch(grid, coordi, 0);
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
	
	float coneRatio = 0.677;//tan(30*3.14159/180.0);
	float maxDist = 1;
	vec4 il=vec4(0), re=vec4(0), shadow=vec4(0);
	
	float sbeta = sin(45*3.14159/180.0);
	float cbeta = cos(45*3.14159/180.0);
	float alpha = 60;
	float alpha2 = 30;
	    il  = voxelConeTrace(coord, normalize(normal), coneRatio, maxDist);
	    il  += 0.5 * voxelConeTrace(coord, normalize(normal*sbeta+tangent*cbeta),  coneRatio, maxDist);
	    il  += 0.5 * voxelConeTrace(coord, normalize(normal*sbeta-tangent*cbeta),  coneRatio, maxDist);
	    il  += 0.5 * voxelConeTrace(coord, normalize(normal*sbeta+tangent*0.5*cbeta + 			bitangent*0.866*cbeta), coneRatio, maxDist);
	    il  += 0.5 * voxelConeTrace(coord, normalize(normal*sbeta+tangent*0.5*cbeta - 			bitangent*0.866*cbeta), coneRatio, maxDist);
	    il  += 0.5 * voxelConeTrace(coord, normalize(normal*sbeta-tangent*0.5*cbeta + 			bitangent*0.866*cbeta), coneRatio, maxDist);
	    il  += 0.5 * voxelConeTrace(coord, normalize(normal*sbeta-tangent*0.5*cbeta - 			bitangent*0.866*cbeta), coneRatio, maxDist);
	// ao += 0.707 * voxelConeTrace(coord, normalize(normal+tangent), coneRatio, maxDist);
	// ao += 0.707 * voxelConeTrace(coord, normalize(normal-tangent), coneRatio, maxDist);z
	// ao += 0.707 * voxelConeTrace(coord, normalize(normal+bitangent), coneRatio, maxDist);
	// ao += 0.707 * voxelConeTrace(coord, normalize(normal-bitangent), coneRatio, maxDist);
	
	//if (shininess > 80.0) 
	//{
		 //vec3 camDir = normalize(coord*2-1 - camPos);//normalize(vec3(-2,2,2));
		 //re += voxelConeTrace(coord, reflect(camDir, normal), 0.001, 1.0);
		//il += voxelConeTrace2(coord, normalize(camDir), 0.25, 1.0);
	//}
	//if (dot(normal, vec3(4.2,10,2)) > 0 && color.a != 0)
	//if (dot(normal, vec3(4.2,10,2)) > 0 )
	//	shadow = voxelConeTrace(coord, normalize(vec3(4.2,10,2)), 0.0001, 1.0);
	//else shadow = vec4(0);
	// shadow.a = max(0.20, shadow.a);
	if (color.a != 0.0)
		outColor = color*1.5;
		//outColor = 0.5* color * color.a + color * il * 0.5 * (1- il.a*.15);
	else
		outColor = (0.05 * color + color *  il * 0.2) *  (1 - il.a*0.25);
	outColor = vec4(outColor.xyz, 1);	
	float level = 1;
	texel = texelFetch(grid, coordi/int(pow(2,level)), int(level));
}