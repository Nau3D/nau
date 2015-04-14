#version 440

in vec3 normalV;
flat in ivec3 posW;

out vec4 outColor;

uniform sampler3D grid;
uniform sampler3D gridN;
uniform mat4 VM;

void main()
{
	if (texelFetch(grid, posW, 0).w == 1.0) {
		vec3 normal = texelFetch(gridN, posW, 0).xyz;
		float intensity = max(0.0, dot(normal, normalize(vec3(1,2,3))));	
		outColor = vec4(texelFetch(grid, posW, 0) * (intensity + 0.2));
	}
	else 
		discard;
	
}