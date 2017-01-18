#version 430

layout(binding=1, offset=0) uniform atomic_uint atP1;   //Light-Facing Pixels (Prime)
layout(binding=1, offset=4) uniform atomic_uint atP2;   //Not Light-Facing Pixels (Prime)
layout(binding=1, offset=8) uniform atomic_uint atP3;   //Pixels in Light (Prime)
layout(binding=1, offset=12) uniform atomic_uint atP4;  //Pixels in Shadow (Prime)
layout(binding=1, offset=16) uniform atomic_uint atP5;  //Total Pixels (Prime)



layout(std430, binding = 3) buffer hitsBuffer {
	vec4 hits[];
};

struct ray {
	vec4 pos;
	vec4 dir;
};

layout(std430, binding = 2) buffer RayBuffer {
	ray rays[];
};

uniform int RenderTargetX;
uniform int RenderTargetY;

in vec2 texCoordV;
out vec4 colorOut;

void main() {
//	ivec2 coord = ivec2(gl_FragCoord.xy);
	ivec2 coord = ivec2(texCoordV*vec2(RenderTargetX,RenderTargetY));
	//int coordB = coord.x + coord.y * RenderTargetX;
	int coordB = coord.x * RenderTargetY + coord.y;
	
	vec4 h = hits[coordB];
	
	//atomicCounterIncrement(atP5); //Total
	if (h.y >= 0.0){
		//atomicCounterIncrement(atP4); //Shadow Optix
		colorOut = vec4(0.0, 0.0, 0.0, 1.0);
	}
	else {
		//atomicCounterIncrement(atP3); //Light Optix
		colorOut = vec4(1.0, 1.0, 1.0, 1.0);
	}	 
//	colorOut = vec4(length(rays[coordB].pos)/100,0,0,1);
//	colorOut = h;
}	