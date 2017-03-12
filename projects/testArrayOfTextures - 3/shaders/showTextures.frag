#version 430
#extension GL_ARB_bindless_texture: enable

layout(rgba8) uniform image2D tex[2];

//layout(std430, binding = 1) buffer texPtr {
//	layout(rgba8) image2D tex[];
//};
//uniform Samplers {
 //       layout(rgba8) image2D tex[2];
  //    };

in vec2 texCoordV;
out vec4 colorOut;

void main() {

	if (texCoordV.x > 0.5) {
		colorOut = imageLoad(tex[0], ivec2(texCoordV * ivec2(128, 128)));
	}
	else {
		colorOut = imageLoad(tex[1], ivec2(texCoordV * ivec2(128, 128)));
	}
}	