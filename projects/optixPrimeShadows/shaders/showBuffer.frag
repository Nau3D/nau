#version 430

layout(std430, binding = 2) buffer hitsBuffer {
	vec4 hits[];
};

struct ray {
	vec4 pos;
	vec4 dir;
};

layout(std430, binding = 1) buffer RayBuffer {
	ray rays[];
};


uniform int size;

in vec2 texCoordV;
out vec4 colorOut;

void main() {
	ivec2 coord = ivec2(texCoordV*vec2(size,size));
	int coordB = coord.x* size + coord.y;

	 vec4 h = hits[coordB];
	
	if (h.y >= 0.0)
		colorOut = vec4(0.2, 0.2, 0.2, 1.0);
	 else
		 colorOut = vec4(1.0, 1.0, 1.0, 1.0);
		 
}	