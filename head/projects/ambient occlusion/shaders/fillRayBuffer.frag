#version 430

#define SAMPLES 9

uniform sampler2D texPos;
uniform sampler2D texNormal;

struct ray {

	// vec3 pos;
	// float tmin;
	// vec3 dir;
	// float tmax;

	vec4 pos;
	vec4 dir;
};

layout(std430, binding = 1) volatile buffer RayBuffer {
	ray rays[];
};

in vec2 texCoordV;

out vec4 outColor;

void main()
{
	vec3 normal = texture(texNormal, texCoordV).xyz * 2.0 - 1.0;
	vec3 pos = texture(texPos, texCoordV).xyz;
	vec3 raios[9], raio;
	raios[0] = vec3(0,2,0);
	raios[1] = vec3(0.5, 0.5, 0.5);
	raios[2] = vec3(-0.5, 0.5, 0.5);
	raios[3] = vec3(0.5, 0.5, -0.5);
	raios[4] = vec3(-0.5, 0.5, -0.5);
	raios[5] = vec3(-0.8, 0.2, 0.0);
	raios[6] = vec3(0.8, 0.2, 0.0);
	raios[7] = vec3(0.0, 0.2, -0.8);
	raios[8] = vec3(0.0, 0.2, 0.8);
	
	ivec2 coord = ivec2(texCoordV.xy*vec2(1024,1024));
	int coordB = coord.x * SAMPLES + coord.y * 1024 * SAMPLES;
	
	vec3 u,k;
	if (dot(normal, vec3(0,1,0)) < 0.9)
		k = vec3(0,1,0);
	else
		k = vec3(1,0,0);
	u = normalize(cross(k, normal));
	vec3 v = cross(normal, u);
	ray r;
	for(int i = 0; i < SAMPLES; ++i){
		
		r.pos = vec4(pos, 0.02);
		raio = raios[i]*5;
		float k = length(raio);
		raio = u * raio.x + normal * raio.y + v * raio.z;
		raio = normalize(raio);
		r.dir = vec4(raio, k/*length(raio)*/);
		
		rays[coordB+i] = r;
		//coordB ++;
	}
	 outColor=vec4(0);
}
