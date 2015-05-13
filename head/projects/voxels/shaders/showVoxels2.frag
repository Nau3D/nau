#version 440

in vec3 normalV;
flat in ivec3 posW;

out vec4 outColor;

uniform sampler3D grid;
uniform sampler3D gridN;
uniform mat4 VM;
uniform int level = 0;

void main()
{

	if (texelFetch(grid, ivec3(posW/(pow(2.0,level))), level).w != 0.0) {
		float intensity = max(0.0, dot(normalV, normalize(vec3(1,2,3))));	
		outColor = vec4(texelFetch(grid, ivec3(posW/(pow(2.0,level))), level) * (intensity + 0.4));
	}
	else 
		discard;
	
}