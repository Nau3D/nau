#version 430

layout(binding=1, offset=276) uniform atomic_uint at22;		//Doubt Pixels(Total)(Apontador)
//layout(binding=1, offset=92) uniform atomic_uint at23;	//Doubt Pixels(Total)(Apontador)


layout(std430, binding = 3) buffer testBuffer {			//vec2 com posição no ecra, e um int = indice do RayBuffer
	vec2 screenPos[];  
};

/* layout(std430, binding = 3) buffer testBuffer {
	vec4 screenPos[];
}; */

uniform int RenderTargetX;
uniform int RenderTargetY;
in vec2 texCoordV;
out vec4 colorOut;

void main() {
//	ivec2 coord = ivec2(gl_FragCoord.xy);	
 	ivec2 coord = ivec2(texCoordV*vec2(RenderTargetX,RenderTargetY));
	int coordB = coord.x* RenderTargetX + coord.y;

	uint p = atomicCounter(at22);
	vec2 i = screenPos[coordB];
	vec2 h = i;
	
	if (h.x == coord.x && h.y == coord.y){
		colorOut = vec4(1.0, 1.0, 1.0, 1.0); 
	}
	else
		colorOut = vec4(0.0, 0.0, 0.0, 0.0);
	
	//colorOut = vec4(h,0,0);
	
/*	if (h.y >= 0.0)
		colorOut = vec4(0.0, 0.0, 0.0, 1.0);
	 else
		 colorOut = vec4(1.0, 1.0, 1.0, 1.0); */
		 
//	colorOut = vec4(length(rays[coordB].pos)/100,0,0,1);
//	colorOut = vec4(rays[coordB].pos);
}	