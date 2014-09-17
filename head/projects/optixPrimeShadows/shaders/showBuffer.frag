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

in vec2 texCoordV;
out vec4 colorOut;

void main() {
//	ivec2 coord = ivec2(gl_FragCoord.xy);
	ivec2 coord = ivec2(texCoordV*vec2(1024,1024));
	int coordB = coord.x* 1024 + coord.y;

	 vec4 h = hits[coordB];
	
	if (h.y >= 0.0)
		colorOut = vec4(0.2, 0.2, 0.2, 1.0);
	 else
		 colorOut = vec4(1.0, 1.0, 1.0, 1.0);
		 
//	colorOut = vec4(length(rays[coordB].pos)/100,0,0,1);
//	colorOut = vec4(rays[coordB].pos);
}	