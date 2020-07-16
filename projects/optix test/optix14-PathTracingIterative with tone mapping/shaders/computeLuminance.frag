#version 430

in vec2 texPos;

layout(std430, binding = 1) buffer accumBuffer {
	vec4 color[];
};

out vec2 luminance;

uniform int res;

void main() {
	
    ivec2 coord = ivec2(texPos * ivec2(res,res));
	int coordB = coord.y* res + coord.x;
    vec4 c2 = color[coordB];

	float lumi = (0.2126*c2.r)+(0.722*c2.g)+(0.07152*c2.b);
	luminance = vec2(lumi*16);
}	