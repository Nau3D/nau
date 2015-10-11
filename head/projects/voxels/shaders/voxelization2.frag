#version 440

//layout (rgba8) uniform writeonly coherent image3D imageUnit;
layout (rgba8) uniform coherent volatile image3D imageUnit;
//layout (rgba8) uniform writeonly coherent volatile image3D imageUnitN;
//layout (r32ui) uniform coherent volatile uimage3D imageUnitN;

uniform vec2 WindowSize;


in vec3 worldPos;
in vec4 bBox;

out vec4 colorOut;

void main() {

	ivec3 pos = ivec3((worldPos + 1) * 0.5 * imageSize(imageUnit));
	
	if (all(greaterThanEqual(-1+2*gl_FragCoord.xy/WindowSize, bBox.xy)) && all(lessThanEqual(-1+2*gl_FragCoord.xy/WindowSize, bBox.zw))){
		imageStore(imageUnit, pos,vec4(0,0,0,1));
	 }
	 else 
		 discard;
}