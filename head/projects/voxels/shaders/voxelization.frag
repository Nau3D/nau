#version 440

layout (rgba8) uniform writeonly volatile image3D imageUnit;

layout(binding=1, offset=0)  uniform atomic_uint at0;

uniform vec2 WindowSize;
in vec4 worldPos;
in vec4 bBox;
in vec4 color;

out vec4 colorOut;

void main() {

	vec4 wp = worldPos / worldPos.w;
	int z = int((worldPos.z + 1) * 0.5 * imageSize(imageUnit).z); 
	int y = int((worldPos.y + 1) * 0.5 * imageSize(imageUnit).y); 
	int x = int((worldPos.x + 1) * 0.5 * imageSize(imageUnit).x); 
	
	//if (all(greaterThanEqual(-1+2*gl_FragCoord.xy/WindowSize, bBox.xy)) && all(lessThanEqual(-1+2*gl_FragCoord.xy/WindowSize, bBox.zw))){
		imageStore(imageUnit, ivec3(x,y,z),color);
		//atomicCounterIncrement(at0);
	// }
	// else 
		// discard;

	colorOut=color;
}