#version 430

struct hit {
	float t;
	int t_id;
	float u,v;
};


layout(std430, binding = 2) buffer hitsBuffer {
	hit hits[];
};

uniform sampler2D normalMap;


void main() {
	
}	