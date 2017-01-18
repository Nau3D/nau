#version 430

layout(std430, binding = 1) buffer vertBuffer {
	int value[];
};

in vec2 texCoordV;
out vec4 colorOut;

void main() {
	ivec2 coord = ivec2(texCoordV * ivec2(256,256));
	int coordB = coord.x* 256 + coord.y;
	int test = value[coordB];
	if (test  %3 == 0)	
		colorOut = vec4(1,0,0,0);
	else if (test % 3 == 1)
		colorOut = vec4(0,1,0,0);
	else
		colorOut = vec4(0,0,1,0);
}	