#version 440

layout (rgba8) uniform writeonly coherent image3D imageUnit;
layout (rgba8) uniform writeonly coherent image3D imageUnitN;

layout(binding=1, offset=0)  uniform atomic_uint at0;

uniform vec2 WindowSize;
uniform vec4 diffuse;
uniform float shininess;
uniform int texCount;
uniform sampler2D texUnit;


in vec4 worldPos;
in vec4 bBox;
in vec4 colorG;
in vec3 normalG;
in vec2 texCoordG;

out vec4 colorOut;

void main() {

	vec4 color;
	if (texCount == 0) 
		color = vec4(diffuse.xyz, 1.0);
	else 
		color = vec4(texture(texUnit, texCoordG).xyz, 1.0);

	vec4 wp = worldPos / worldPos.w;
	int z = int((worldPos.z + 1) * 0.5 * imageSize(imageUnit).z); 
	int y = int((worldPos.y + 1) * 0.5 * imageSize(imageUnit).y); 
	int x = int((worldPos.x + 1) * 0.5 * imageSize(imageUnit).x); 
	
	if (all(greaterThanEqual(-1+2*gl_FragCoord.xy/WindowSize, bBox.xy)) && all(lessThanEqual(-1+2*gl_FragCoord.xy/WindowSize, bBox.zw))){
		imageStore(imageUnit, ivec3(x,y,z),color);
		imageStore(imageUnitN, ivec3(x,y,z), vec4(normalG * 0.5 + 0.5,shininess));
		atomicCounterIncrement(at0);
	 }
	 else 
		 discard;

	colorOut=color;
}