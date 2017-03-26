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

vec4 voxelConeTrace1(vec3 origin, vec3 dir, float coneRatio, float maxDist) {

	vec3 samplePos = origin;
	float accum = 0.0;
	
	float minDiameter = 2.0/GridSize;
	
	float startDist = 2 * minDiameter;
	float dist = startDist;
	
	while(dist < maxDist && accum < 1.0) {
	
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
		sampleValue.a = 1.0 - pow(1.0 - sampleValue.a, minDiameter/sampleDiameter);
		//vec4 sampleValue = texelFetch(grid, ivec3(samplePos*512/pow(2.0,level)), level);	
		//sampleValue.a /= minDiameter/sampleDiameter;
		if (dist < 0.01)
			accum += sampleValue.a;// * minDiameter/sampleDiameter ;
		// if (level > 2.0)
			// accum = 0.5;
			
		// else accum = 0.0;	
		}
		dist += sampleDiameter/2.0;
		//dist += 1000;
	}
	return vec4(accum);
}

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
		
		//if (sampleLOD >= 5 && sampleLOD < 6) 
		{
		vec4 sampleValue = textureLod(grid, samplePos , level);
		vec3 sampleNormal = textureLod(gridN, samplePos, level).xyz * 2 - 1;
		float sampleWeight = max(0.0, -dot(sampleNormal, dir))*(1- sampleValue.a);
		// if (dot(sampleNormal, dir) > 0.0)
			// sampleWeight = 0;
		// else	
			// sampleWeight = (1.0 - accum.a);
		//accum.a += sampleValue.a * sampleWeight;
//sampleValue.a ;
//		accum.rgb += sampleValue.rgb ;//* sampleValue.a;
		//sampleValue.a = 1.0 - pow(1.0 - sampleValue.a, minDiameter/sampleDiameter);
		
		//ao += sampleValue.a;
		sampleValue.a /= minDiameter/sampleDiameter;
		
		// Either
		//accum.rgb += sampleValue.rgb * (sampleValue.a) * (1.0 - accum.a);//* 		// if (level > 2.0)
		//if (dot(sampleNormal, dir) < 0.0)
			 // accum.rgb += sampleValue.rgb * sampleValue.a * (1-accum.a);		
		accum.rgb += sampleValue.rgb  * (1-accum.a);		
		accum.a += sampleValue.a;//sampleValue.a;// * minDiameter/sampleDiameter ;
	// if (level > 2.0)
			// accum = 0.5;
			
		// else accum = 0.0;	
		}
		dist += sampleDiameter*2;
		//dist += 0.5;
	}
	accum.a = ao;
	return accum;
}

vec4 voxelConeTrace2(vec3 origin, vec3 dir, float coneRatio, float maxDist) {

	vec3 samplePos = origin;
	vec4 accum = vec4(0.0);
	float ao = 0;
	float minDiameter = 2.0/GridSize;
	
	float startDist = 2 * minDiameter;
	float dist = startDist;
	
	while(dist < maxDist && accum.a < 1.0) {
	
		float sampleDiameter = max(minDiameter, coneRatio * dist);
		//sampleDiameter += minDiameter;
		
		// convert diameter to LOD
		// for example:
		// -log2(1/256) = 8
		// -log2(1/128) = 7
		// -log2(1/64) = 6
		float sampleLOD = -log2(sampleDiameter);
		
		vec3 samplePos = origin + dir * (dist);
		float level =  max(0, log2(GridSize) - sampleLOD  );
		//level = 0;
		//if (sampleLOD >= 5 && sampleLOD < 6) 
		{
		vec4 sampleValue = textureLod(grid, samplePos , level);
		vec3 sampleNormal = textureLod(gridN, samplePos, level).xyz * 2 - 1;
		float sampleWeight = max(0.0, -dot(sampleNormal, dir))*(1- sampleValue.a);
		// if (dot(sampleNormal, dir) > 0.0)
			// sampleWeight = 0;
		// else	
			// sampleWeight = (1.0 - accum.a);
		//accum.a += sampleValue.a * sampleWeight;
//sampleValue.a ;
//		accum.rgb += sampleValue.rgb ;//* sampleValue.a;
		//sampleValue.a = 1.0 - pow(1.0 - sampleValue.a, minDiameter/sampleDiameter);
		if (dist < 0.01)
			ao += 1.0 - pow(1.0 - sampleValue.a, minDiameter/sampleDiameter);
		sampleValue.a /= minDiameter/sampleDiameter;
		
		// Either
		//accum.rgb += sampleValue.rgb * (sampleValue.a) * (1.0 - accum.a);//* 		// if (level > 2.0)
		//if (dot(sampleNormal, dir) < 0.0)
			 // accum.rgb += sampleValue.rgb * sampleValue.a * (1-accum.a);		
			accum.rgb += sampleValue.rgb  * (1-accum.a);		
		accum.a += sampleValue.a;//sampleValue.a;// * minDiameter/sampleDiameter ;
	// if (level > 2.0)
			// accum = 0.5;
			
		// else accum = 0.0;	
		}
		dist += sampleDiameter;
		//dist += 0.5;
	}
	accum.a = ao;
	return accum;
}

void main()
{
	// fetch color from render to texture
	vec4 color = texture(texColor, texCoordV);
	
	if (color.a != 0.0) {
		outColor = color*1.5;
		return;
	}
		
	// fetch world position	
	vec3 coord = texture(texPos, texCoordV).xyz;
	// convert position to grid index
	ivec3 coordi = ivec3(coord * GridSize);
	// fetch voxel from grid
	vec4 texel = texelFetch(grid, coordi, 0);
	// fetch normal for pixel
	vec3 normal = texture(texNormal, texCoordV).xyz ;
	normal = normalize(normal);
	// build reference system
	vec3 tangent, bitangent;
	vec3 c1 = cross(normal, vec3(0,0,1));
	vec3 c2 = cross(normal, vec3(0,1,0));
	if (length(c1) > length(c2)) 
		tangent = c1;
	else 
		tangent = c2;
	tangent = normalize(tangent);
	bitangent = cross(normal, tangent);
	
	// trace cones cones
	float coneRatio = 0.677;//tan(30*3.14159/180.9);
	float maxDist = 1;
	vec4 il=vec4(0), re=vec4(0), shadow=vec4(0);
	
	float sbeta = sin(45);
	float cbeta = cos(45);
	float alpha = 60;
	float alpha2 = 30;
    il  = voxelConeTrace(coord, normalize(normal), coneRatio, maxDist);
    il  += 0.5 * voxelConeTrace(coord, normalize(normal*sbeta+tangent*cbeta),  coneRatio, maxDist);
    il  += 0.5 * voxelConeTrace(coord, normalize(normal*sbeta-tangent*cbeta),  coneRatio, maxDist);
    il  += 0.5 * voxelConeTrace(coord, normalize(normal*sbeta+tangent*0.5*cbeta + 					bitangent*0.866*cbeta), coneRatio, maxDist);
    il  += 0.5 * voxelConeTrace(coord, normalize(normal*sbeta+tangent*0.5*cbeta - 					bitangent*0.866*cbeta), coneRatio, maxDist);
    il  += 0.5 * voxelConeTrace(coord, normalize(normal*sbeta-tangent*0.5*cbeta + 					bitangent*0.866*cbeta), coneRatio, maxDist);
    il  += 0.5 * voxelConeTrace(coord, normalize(normal*sbeta-tangent*0.5*cbeta - 					bitangent*0.866*cbeta), coneRatio, maxDist);

	outColor = (0.05 * color + color *  il * 0.2) * (1- il.a*0.25);
	outColor = vec4(outColor.xyz, 1);	
	return;
		// ao += 0.707 * voxelConeTrace(coord, normalize(normal+tangent), coneRatio, maxDist);
	// ao += 0.707 * voxelConeTrace(coord, normalize(normal-tangent), coneRatio, maxDist);z
	// ao += 0.707 * voxelConeTrace(coord, normalize(normal+bitangent), coneRatio, maxDist);
	// ao += 0.707 * voxelConeTrace(coord, normalize(normal-bitangent), coneRatio, maxDist);
	
	//if (shininess > 80.0) 
	//{
		 vec3 camDir = normalize(coord*2-1 - camPos);//normalize(vec3(-2,2,2));
		 //re += voxelConeTrace(coord, reflect(camDir, normal), 0.001, 1.0);
		//il += voxelConeTrace2(coord, normalize(camDir), 0.25, 1.0);
	//}
	//if (dot(normal, vec3(4.2,10,2)) > 0 && color.a != 0)
	if (dot(normal, vec3(4.2,10,2)) > 0 )
		shadow = voxelConeTrace(coord, normalize(vec3(4.2,10,2)), 0.0001, 1.0);
	else shadow = vec4(0);
	// shadow.a = max(0.20, shadow.a);
	if (color.a != 0.0)
		outColor = color*1.5;
		//outColor = 0.5* color * color.a + color * il * 0.5 * (1- il.a*.15);
	else
		outColor = (0.05 * color + color *  il * 0.2) * (1- il.a*0.25);
	outColor = vec4(outColor.xyz, 1);	
	// if (color.a != 0.0)
		// outColor = color*1.5;
	// else
		// outColor = vec4(0.0);
		
	//outColor *= 1.5;	
	// outColor = vec4(1-il.a*0.25);	
	// outColor = vec4(1-il.a*0.25);
	//outColor = color;//* color.a;	
	//  outColor =  il*0.2;// color *  vec4(1- il.a*0.10);
	//outColor = vec4((color * il * 0.33 )*(1- il.a*0.33)) ;
	//outColor = vec4((color * il*0.33)*(1- il.a*0.10)) ;
	//outColor = vec4(1- il.a*0.25);
	 // outColor = re;
	// outColor = re;
	 // outColor = vec4(1- shadow.a);
	//outColor = il*0.5;//vec4(1-il.a*0.15);//vec4(normal*0.5 + 0.5,0);
	float level = 0;
	texel = texelFetch(grid, coordi/int(pow(2,level)), int(level));
	//float a = 1.0 - pow(1.0 - texel.a,255);
	//outColor = texel;
	//  outColor = vec4(texel.a);
	//outColor =  color;//vec4(normal,1);
	//outColor = vec4(1-shadow.a);
}