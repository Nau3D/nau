#version 430

struct hit {
	float t;
	int t_id;
	float u,v;
};

layout(binding=1, offset=0) uniform atomic_uint atP1;   //Light-Facing Pixels (Prime)
layout(binding=1, offset=4) uniform atomic_uint atP2;   //Not Light-Facing Pixels (Prime)
layout(binding=1, offset=8) uniform atomic_uint atP3;   //Pixels in Light (Prime)
layout(binding=1, offset=12) uniform atomic_uint atP4;  //Pixels in Shadow (Prime)
layout(binding=1, offset=16) uniform atomic_uint atP5;  //Total Pixels (Prime)



layout(std430, binding = 2) buffer hitsBuffer {
	hit hits[];
//	vec4 hits[];
};

uniform sampler2D texColor;
uniform sampler2D texNormal;
uniform vec3 lightDirection;
uniform int RenderTargetX;
uniform int RenderTargetY;
uniform mat4 V;

in vec2 texCoordV;
out vec4 colorOut;

void main() {
	ivec2 coord = ivec2(texCoordV * ivec2(RenderTargetX,RenderTargetY));
	//int coordB = coord.x + coord.y * RenderTargetX;
	int coordB = coord.x * RenderTargetY + coord.y;
	
	 hit h = hits[coordB];
	
	vec4 color = texture(texColor, texCoordV);
	bool pvcl = (color.r==0.0 && color.g==0.0 && color.r==0.0);
	
	atomicCounterIncrement(atP5); //Total
	if (pvcl) {
		atomicCounterIncrement(atP2); //PVCL Optix
		colorOut = vec4(0.0,1.0,0.0,1.0);
	
	}
	else {
		atomicCounterIncrement(atP1); //PVPL Optix
		 if (h.t_id >= 0){
			atomicCounterIncrement(atP4); //Shadow Optix
			colorOut = color * vec4(0.0, 0.0, 0.0, 1.0);
			
		 }
		 else{
			 atomicCounterIncrement(atP3); //Light Optix
			 colorOut = color * vec4(1.0, 1.0, 1.0, 1.0);
		}
	}		
}	


/*
	vec4 color = texture(texColor, texCoordV);
	vec3 n = texture(texNormal, texCoordV).xyz * 2.0 - 1.0;

	vec3 ld = vec3(V * vec4(-lightDirection, 0.0));
	float intensity = max(0.0, dot(n, normalize(ld)));
	color = vec4(color.rgb*intensity+0.2, color.a);
	
	 if (h.t_id >= 0)
		colorOut = color * vec4(0.5, 0.5, 0.5, 1.0);
	 else
		 colorOut = color * vec4(1.0, 1.0, 1.0, 1.0);


*/