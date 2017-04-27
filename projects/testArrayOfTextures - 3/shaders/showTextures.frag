#version 430
#extension GL_ARB_bindless_texture: enable

layout(rgba8) uniform image2D tex1[2];
layout(rgba8) uniform image2D tex2[2];

in vec2 texCoordV;
out vec4 colorOut;

void main() {

	if (texCoordV.x < 0.5) {
		colorOut = imageLoad(tex1[0], ivec2(texCoordV * ivec2(128, 128)));
	}
	else {
		colorOut = imageLoad(tex2[1], ivec2(texCoordV * ivec2(128, 128)));
	}
}	