#version 430 

layout(binding=2, offset=0)  uniform atomic_uint at0;
layout(binding=2, offset=4) uniform atomic_uint at1;
layout(binding=2, offset=8) uniform atomic_uint at2;
layout(binding=2, offset=12) uniform atomic_uint at3;

uniform sampler2D texUnit;

in vec4 texCoord;

out vec4 outColor;

void main()
{
	vec4 color = texture(texUnit, texCoord.xy);
	
	float m = max(max(color.r, color.g),color.b);
	
	if (m == color.r) {
		atomicCounterIncrement(at0);
	}
	else 	if (m == color.g) {
		 atomicCounterIncrement(at1);
	 }
     else
		 atomicCounterIncrement(at2);
		 
	atomicCounterIncrement(at3);
	memoryBarrier(); 
	outColor = color;

		
}
