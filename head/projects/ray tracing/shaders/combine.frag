#version 430

in vec2 texPos;

uniform sampler2D tex2;
uniform uint frameCount;

layout(binding = 1, rgba8) uniform image2D tex1;

out vec4 outColor;

void main() {
	
	ivec2 imageCoords = ivec2(texPos * vec2(1024,1024));
	vec4 c1 = imageLoad(tex1, imageCoords);
	vec4 c2 = texture(tex2, texPos);
	
	vec4 cFinal = (c1 * (frameCount+1) + c2) / (frameCount+2);
	
	memoryBarrier();
	imageStore(tex1, imageCoords, cFinal);
}	