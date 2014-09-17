#version 430

layout(std430, binding = 1) buffer testBuffer {
	vec4 color[];
};

in vec2 texCoordV;
out vec4 colorOut;

void main() {
	ivec2 coord = ivec2(texCoordV * ivec2(512,512));
	int coordB = coord.x* 512 + coord.y;

	colorOut = color[coordB];
	// if (coord.y  %3 == 0)	
		// colorOut = vec4(1,0,0,0);
	// else if (coord.y % 3 == 1)
		// colorOut = vec4(0,1,0,0);
	// else
		// colorOut = vec4(0,0,1,0);
}	