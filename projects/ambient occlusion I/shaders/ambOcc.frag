#version 330

#define SAMPLES 1

uniform sampler2D depth;
uniform sampler2D normals;
uniform sampler2D positions;

uniform mat4 CamP;
uniform mat4 CamV;


in vec4 texPos;

layout (location = 0) out vec4 occMap;

float dist = 0.05;

void main(void) {

	vec3 rays[9], ray;
	rays[0]	= vec3(0,2,0);
	rays[1] = vec3(0.5, 0.5, 0.5);
	rays[2] = vec3(-0.5, 0.5, 0.5);
	rays[3] = vec3(0.5, 0.5, -0.5);
	rays[4]	= vec3(-0.5, 0.5, -0.5);
	rays[5] = vec3(-0.5, 0.1, 0.5);
	rays[6] = vec3(0.5, 0.1, -0.5);
	rays[7] = vec3(-0.5, 0.1, -0.5);
	rays[8] = vec3(0.5, 0.1, 0.5);

	float occ = 0.0;
	vec4 pos = texture(positions, texPos.xy);
	float depthV = texture(depth, texPos.xy).r;
	vec3 normal =  texture(normals,texPos.xy).xyz;// * 2.0 -1.0;
	float hasGeometry = texture(normals,texPos.xy).w;
	vec3 u = cross(normal, vec3(0,1,0));
	if (dot(u,normal) < 1.e-3f)
		u = cross(normal, vec3(1,0,0));
	u = normalize(u);
	vec3 v = cross(normal, u);
	
	for (int i = 0; i < 9; ++i) {
	
		ray = normalize(rays[i]); // just to test
		//ray = vec3(CamV * vec4(ray,0));
		ray = u * ray.x + normal * ray.y + v * ray.z;
		// ray.x = ray.x * u.x + ray.x * normal.x + ray.x * v.x;
		// ray.y = ray.y * u.y + ray.y * normal.y + ray.y * v.y;
		// ray.z = ray.z * u.z + ray.z * normal.z + ray.z * v.z;
		vec4 samplePos = pos + dist * vec4(ray,0); // in camera space
		vec4 sampleProjPos = CamP * samplePos; // in clip space
		sampleProjPos = sampleProjPos/sampleProjPos.w;
		sampleProjPos = sampleProjPos * 0.5 + 0.5;
		float sampleProjDepth = sampleProjPos.z;
		
		float sampleRecordedDepth = texture(depth, sampleProjPos.xy).r;
		
		if ((sampleRecordedDepth > sampleProjDepth) || !(sampleProjPos.x >= 0 && sampleProjPos.y >= 0 && sampleProjPos.x < 1 && sampleProjPos.y < 1))
				++occ;
		
	}
	occMap = vec4(occ/9.0);
	if (hasGeometry	== 0)
		occMap = vec4(0);
	else if (occ == 0)
		occMap = vec4(0.25);
	else
		occMap = vec4(occ/9.0);
		
	//occMap = vec4(	sampleProjDepth);
}
/*	vec4 currentPixel = texture(normals,texPos.xy);
	float currentPixelDepth = texture(depth,texPos.xy).x;
	vec4 vertexPos = texture(positions, texPos.xy);
	float sampleDepth;
	vec4 sampleProj;
	float occDepth;
	float occ = 0.0;
	for(int i=0; i<1; ++i) {
		
		//create ray    TODO: criar raio
		vec4 ray = vec4(0.f, 1.0f, 0.f, 0.0f);
		
		ray = CamV * ray;
		vec4 sample1 = vertexPos + ray; // para obter o vertexPos multiplicar por VIEW_MODEL
		vec4 sample2 = CamP * vertexPos;
		sample2 = sample2/sample2.w * 0.5 + 0.5;
		sampleProj = (CamP * sample1);// * 0.5 + 0.5; // CamP é a projection matrix
		sampleProj = sampleProj/sampleProj.w * 0.5 + 0.5;
		sampleDepth = length(sample1); //length(texture(positions,sample2.xy).xyz);//sampleProj.z; 
		sampleDepth = length(texture(positions,sample2.xy).xyz);//sampleProj.z; 
		if (sampleProj.x >= 0 && sampleProj.y >= 0 && sampleProj.x < 1 && sampleProj.y < 1)
			occDepth = length(texture(positions,sampleProj.xy).xyz);		
		else
			occDepth= 0;
/*		if (sampleProj.x >= 0 && sampleProj.y >= 0 && sampleProj.x < 1 && sampleProj.y < 1)
			occDepth = texture(depth,sampleProj.xy).x;		
		else
			occDepth= 0;
*/		
		//get depth difference
	//	float depthDiff = sampleDepth - occDepth; // testar sinais
		
		
/*		
		//position of intersect 
		vec4 rayProj = (CamM * M * ray) * 0.5 + 0.5;
		vec4 sample = vertexPos + ray;//sign(dot(ray,currentPixel.xyz))*ray;
		vec4 sampleDepth = texture(depth, sample.xy);
		//get occluder depth and normal
		vec3 occNormal = texture2D(normals,sample.xy).xyz;
		float occDepth = texture2D(depth,sample.xy).x;		
	
		//get depth difference
		float depthDiff = sampleDepth - occDepth; // testar sinais
		*/	
		//increment occ factor //TODO: verificiar se calculos estao certos
		/*if(depthDiff <= 0)
			occ = 1.f;
		else
			occ = 0.5f;//+= step(0.f, depthDiff);
	}
	
	//occMap = vec4((occ/SAMPLES));//occ / SAMPLES;
	//if (occDepth == 1)
		//occDepth = 0;
	//if (texture(normals, sampleProj.xy).w == 1)
	if (occDepth == 0)
		occMap = vec4(0);
	else	
		occMap = vec4(occ);
	//else
}*/