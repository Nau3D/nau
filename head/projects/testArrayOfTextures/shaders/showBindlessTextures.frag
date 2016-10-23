#version 430

#extension GL_ARB_bindless_texture: enable

layout(std430, binding = 1) buffer texPtr {
	sampler2D tex[];
};

in vec2 texCoordV;
out vec4 colorOut;

void main() {

	sampler2D s;
	if (texCoordV.x > 0.5) {
		s = sampler2D(tex[0]);
	}
	else {
		s = sampler2D(tex[1]);
	}
	colorOut = texture(s, texCoordV);
}	