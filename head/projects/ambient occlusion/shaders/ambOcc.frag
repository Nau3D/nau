#version 430

#define SAMPLES 9

uniform sampler2D depth;
uniform sampler2D normals;
uniform sampler2D positions;

uniform mat4 CamP;
uniform mat4 CamV;
//
struct ray {
	vec4 pos;
	vec4 dir;
};

layout(std430, binding = 1) buffer RayBuffer {
	ray rays[];
};
//

in vec4 texPos;

layout (location = 0) out vec4 occMap;
layout (location = 1) out vec4 occMap2;


void main(void) {


	float hasGeometry = texture(normals,texPos.xy).w;

	vec3 raios[9], raio;
	raios[0] = vec3(0,2,0);
	raios[1] = vec3(0.5, 0.5, 0.5);
	raios[2] = vec3(-0.5, 0.5, 0.5);
	raios[3] = vec3(0.5, 0.5, -0.5);
	raios[4] = vec3(-0.5, 0.5, -0.5);
	raios[5] = vec3(-0.5, 0.1, 0.5);
	raios[6] = vec3(0.5, 0.1, -0.5);
	raios[7] = vec3(-0.5, 0.1, -0.5);
	raios[8] = vec3(0.5, 0.1, 0.5);
	
	vec4 posT;

	
	//index buffer
	
	
	// vec4 posT = texture(positions, texPos.xy);
	// float depthV = texture(depth, texPos.xy);
	// 
	// float hasGeometry = texture(normals,texPos.xy).w;
	// vec3 u = cross(normal, vec3(0,1,0));
	// if (dot(u,normal) < 1.e-3f)
	    // u = cross(normal, vec3(1,0,0));
	// u = normalize(u);
	// vec3 v = cross(normal, u);
	
	ivec2 coord = ivec2(texPos.xy*vec2(1024,1024));
	int coordB = coord.x * SAMPLES + coord.y * 1024 * SAMPLES;
	float occ1 = 0.0;
	for (int i = 0; i < SAMPLES; ++i) {
	
		raio = rays[coordB+i].dir.xyz;
		raio = vec3(CamV * vec4(raio, 0.0));
		raio = normalize(raio)*1.0f; // just to test
		//raio = u * raio.x + normal * raio.y + v * raio.z;
		posT = vec4(rays[coordB+i].pos.xyz,1);
		posT = CamV * posT;
		vec4 samplePos = posT + vec4(raio,0); // in camera space
		vec4 sampleProjPos = CamP * samplePos;
		sampleProjPos = sampleProjPos/sampleProjPos.w;
		sampleProjPos = sampleProjPos * 0.5 + 0.5;
		float sampleProjDepth = sampleProjPos.z;
		
		float sampleRecordedDepth = texture(depth, sampleProjPos.xy).r;
		
		if ((sampleRecordedDepth > sampleProjDepth) || !(sampleProjPos.x >= 0 && sampleProjPos.y >= 0 && sampleProjPos.x < 1 && sampleProjPos.y < 1))
				++occ1;
	}
	//posT = CamV * vec4(rays[coordB].pos.xyz,1);
	if (hasGeometry > 0 )
		occMap2 = vec4(occ1/SAMPLES);
//		occMap2 = posT;	
	else
		occMap2 = vec4(0);

		float occ = 0.0;
	posT = texture(positions, texPos.xy);
	posT = CamV * posT;
	vec3 normal =  texture(normals,texPos.xy).xyz * 2.0 -1.0;
	vec3 n = vec3(CamV * vec4(normal,0));
	vec3 u = cross(n, vec3(0,1,0));
	if (dot(u,n) < 1.e-3f)
		u = cross(n, vec3(1,0,0));
	u = normalize(u);
	vec3 v = cross(n, u);		
	for (int i = 0 ; i < SAMPLES; ++i) {
		raio = normalize(raios[i])*1.0f; // just to test
		//ray = vec3(CamV * vec4(ray,0));
		raio = u * raio.x + n * raio.y + v * raio.z;
		vec4 samplePos = posT + vec4(raio,0); // in camera space
		vec4 sampleProjPos = CamP * samplePos;
		sampleProjPos = sampleProjPos/sampleProjPos.w;
		sampleProjPos = sampleProjPos * 0.5 + 0.5;
		float sampleProjDepth = sampleProjPos.z;
		
		float sampleRecordedDepth = texture(depth, sampleProjPos.xy).r;
		
		if ((sampleRecordedDepth > sampleProjDepth) || !(sampleProjPos.x >= 0 && sampleProjPos.y >= 0 && sampleProjPos.x < 1 && sampleProjPos.y < 1))
				++occ;
		
	}
	if (hasGeometry > 0 )
		occMap = vec4(occ/SAMPLES);
		//occMap = posT;
	else
		occMap = vec4(0);
		

}
