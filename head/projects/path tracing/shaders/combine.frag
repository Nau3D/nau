#version 430

in vec2 texPos;


uniform sampler2D tex2;
uniform uint frameCount;

layout(binding = 1, rgba32f) uniform image2D tex1;

out vec4 outColor;

void main() {
	
	ivec2 imageCoords = ivec2(texPos * textureSize(tex2,0));
	vec4 c1 = imageLoad(tex1, imageCoords);
	vec3 c2 = texture(tex2, texPos).rgb;
	 c2 *=1;
	// Reinhardt
	// c2 = c2/(c2+1);
	// c2 = pow(c2, vec3(1/2.2));
	//Jim Heji
	// c2 = max(vec3(0), c2-0.004);
	// c2 = (c2 * (6.2 * c2 + 0.5))/(c2*(6.2*c2 + 1.7)+ 0.06);
	
	vec4 c3 = vec4(c2,1);
	vec4 cFinal = (c1 * (frameCount-1) + c3) / (frameCount);
	
	memoryBarrier();
	imageStore(tex1, imageCoords, cFinal);
}	