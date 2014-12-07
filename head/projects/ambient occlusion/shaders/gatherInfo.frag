#version 430 

#define SAMPLES 9

in vec3 normalVar;
in vec4 posVar;
in vec4 texPos;

layout (location = 0) out vec4 normalMap;
layout (location = 1) out vec4 posMap;

//buffer
struct ray {
	vec4 pos;
	vec4 dir;
};

layout(std430, binding = 1) buffer RayBuffer {
	ray rays[];
};

void main(void) {
		
	vec3 normal = normalize(normalVar);
	normalMap = vec4((vec3(normal))* 0.5 + 0.5, 1.0);
	posMap = posVar;
	//buffer	
	
	/*vec3 kernel[SAMPLES];
	for (int i = 0; i < SAMPLES; ++i) {
		kernel[i] = vec3(
		random(-1.0f, 1.0f),
		random(-1.0f, 1.0f),
		random(0.0f, 1.0f));
		
		kernel[i] = normalize(kernel[i]);
}*/
	
/*	vec3 raios[9], raio;
	raios[0] = vec3(0,2,0);
	raios[1] = vec3(0.5, 0.5, 0.5);
	raios[2] = vec3(-0.5, 0.5, 0.5);
	raios[3] = vec3(0.5, 0.5, -0.5);
	raios[4] = vec3(-0.5, 0.5, -0.5);
	raios[5] = vec3(-0.5, 0.1, 0.5);
	raios[6] = vec3(0.5, 0.1, -0.5);
	raios[7] = vec3(-0.5, 0.1, -0.5);
	raios[8] = vec3(0.5, 0.1, 0.5);
	
	ivec2 coord = ivec2(texPos.xy*vec2(1024,1024));
	int coordB = coord.x * SAMPLES + coord.y * 1024 * SAMPLES;
	
	vec3 u = cross(normal, vec3(0,1,0));
	if (dot(u,normal) < 1.e-3f)
		u = cross(normal, vec3(1,0,0));
	u = normalize(u);
	vec3 v = cross(normal, u);
	ray r;
	for(int i = 0; i < SAMPLES; ++i){
		
		r.pos = vec4(posVar.xyz, 0.01);
		raio = normalize(raios[i]);
		raio = u * raio.x + normal * raio.y + v * raio.z;
		r.dir = vec4(raio, 5.0);
		
		rays[coordB+i] = r;
		//coordB ++;
	}
*/	
}
