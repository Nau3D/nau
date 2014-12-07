#version 430

#define SAMPLES 9


in vec4 texPos;


struct hit {
	float t;
	int t_id;
	float u,v;
};


layout(std430, binding = 2) buffer hitsBuffer {
	hit hits[];
};

uniform sampler2D normalMap;

layout (location = 0) out vec4 optixTex;

void main() {
	
	ivec2 coord = ivec2(texPos.xy*vec2(1024,1024));
	int coordB = coord.x * SAMPLES + coord.y * 1024 * SAMPLES;

	
	
	float occ=SAMPLES;
	for(int i = 0; i < SAMPLES; i++){
		if (hits[coordB].t_id >= 0)
			--occ;
		++coordB;
	}
	occ /= SAMPLES;
	optixTex = vec4(occ);
	/*if(occ>0)
		optixTex = vec4(vec3(255.f), 1.0);
	else
		optixTex = vec4(vec3(0.f), 1.0);*/
}	